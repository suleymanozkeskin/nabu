use fs2::FileExt;
use rayon::prelude::*;
use regex::{Captures, Regex};
use rusqlite::types::Value as SqlValue;
use rusqlite::{params, params_from_iter, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::env;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Stdio};
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(feature = "semantic")]
use std::sync::Arc;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration as StdDuration, Instant, SystemTime, UNIX_EPOCH};
use time::format_description::well_known::Rfc3339;
use time::{Date, Month, OffsetDateTime};

pub const SCHEMA_VERSION: u32 = 1;
pub const SQLITE_SCHEMA: &str = include_str!("../schema.sql");
pub const MAX_INLINE_ENVELOPE_BYTES: usize = 16 * 1024 * 1024;
const EVENTS_FTS_SCHEMA: &str = r#"
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
const MAX_SEARCH_LIMIT: usize = 50;
const MAX_SEARCH_SNIPPET_CHARS: usize = 1000;
const DEFAULT_SEARCH_SNIPPET_CHARS: usize = 240;
const MAX_SESSION_LIMIT: usize = 500;
const MAX_CONTEXT_EVENTS_PER_SIDE: usize = 500;
const SEMANTIC_MODEL_ID: &str = "embeddinggemma-300m-q4";
const SEMANTIC_MODEL_REPO: &str = "onnx-community/embeddinggemma-300m-ONNX";
const SEMANTIC_VECTOR_DIMENSIONS: usize = 256;
const SEMANTIC_MODEL_REMOTE_FILES: &[(&str, &str)] = &[
    ("onnx/model_q4.onnx", "onnx/model_q4.onnx"),
    ("onnx/model_q4.onnx_data", "onnx/model_q4.onnx_data"),
    ("tokenizer.json", "tokenizer.json"),
    ("config.json", "config.json"),
    ("special_tokens_map.json", "special_tokens_map.json"),
    ("tokenizer_config.json", "tokenizer_config.json"),
];
#[cfg(any(feature = "semantic", test))]
const EMBEDDING_GEMMA_QUERY_PREFIX: &str = "task: search result | query: ";
#[cfg(any(feature = "semantic", test))]
const EMBEDDING_GEMMA_DOCUMENT_PREFIX: &str = "title: none | text: ";
#[cfg(feature = "semantic")]
const SEMANTIC_EMBED_MAX_LENGTH: usize = 2048;
#[cfg(feature = "semantic")]
const SEMANTIC_EMBED_BATCH_SIZE: usize = 64;
#[cfg(feature = "semantic")]
const SEMANTIC_EMBED_WRITE_CHUNK_SIZE: usize = 2048;
#[cfg(feature = "semantic")]
const SEMANTIC_EMBED_COLLECT_BATCH_SIZE: usize = 4096;
#[cfg(feature = "semantic")]
const SEMANTIC_EMBED_PROGRESS_INTERVAL: StdDuration = StdDuration::from_secs(2);

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("validation error: {0}")]
    Validation(String),
    #[error("home directory could not be resolved; set --home or NABU_HOME")]
    HomeUnavailable,
    #[error("filesystem error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("sqlite error at {path}: {source}")]
    Sqlite {
        path: PathBuf,
        #[source]
        source: rusqlite::Error,
    },
    #[error("semantic search unavailable: {0}")]
    SemanticUnavailable(String),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("time formatting error: {0}")]
    TimeFormat(#[from] time::error::Format),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tool {
    Codex,
    Claude,
    Opencode,
}

impl Tool {
    pub fn as_str(self) -> &'static str {
        match self {
            Tool::Codex => "codex",
            Tool::Claude => "claude",
            Tool::Opencode => "opencode",
        }
    }

    pub const fn all() -> [Tool; 3] {
        [Tool::Codex, Tool::Claude, Tool::Opencode]
    }
}

impl fmt::Display for Tool {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for Tool {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "codex" => Ok(Tool::Codex),
            "claude" => Ok(Tool::Claude),
            "opencode" => Ok(Tool::Opencode),
            _ => Err(Error::Validation(format!("unsupported tool: {value}"))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Source {
    Hook,
    EventStream,
    TranscriptTail,
    SdkSessionStore,
    Backfill,
    ExecJson,
    AppServer,
}

impl Source {
    pub fn as_str(self) -> &'static str {
        match self {
            Source::Hook => "hook",
            Source::EventStream => "event_stream",
            Source::TranscriptTail => "transcript_tail",
            Source::SdkSessionStore => "sdk_session_store",
            Source::Backfill => "backfill",
            Source::ExecJson => "exec_json",
            Source::AppServer => "app_server",
        }
    }
}

impl FromStr for Source {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "hook" => Ok(Source::Hook),
            "event_stream" => Ok(Source::EventStream),
            "transcript_tail" => Ok(Source::TranscriptTail),
            "sdk_session_store" => Ok(Source::SdkSessionStore),
            "backfill" => Ok(Source::Backfill),
            "exec_json" => Ok(Source::ExecJson),
            "app_server" => Ok(Source::AppServer),
            _ => Err(Error::Validation(format!("unsupported source: {value}"))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CanonicalType {
    #[serde(rename = "session.started")]
    SessionStarted,
    #[serde(rename = "session.resumed")]
    SessionResumed,
    #[serde(rename = "session.ended")]
    SessionEnded,
    #[serde(rename = "user.message")]
    UserMessage,
    #[serde(rename = "assistant.delta")]
    AssistantDelta,
    #[serde(rename = "assistant.message")]
    AssistantMessage,
    #[serde(rename = "tool.call")]
    ToolCall,
    #[serde(rename = "tool.result")]
    ToolResult,
    #[serde(rename = "permission.requested")]
    PermissionRequested,
    #[serde(rename = "permission.replied")]
    PermissionReplied,
    #[serde(rename = "file.changed")]
    FileChanged,
    #[serde(rename = "compaction.before")]
    CompactionBefore,
    #[serde(rename = "compaction.after")]
    CompactionAfter,
    #[serde(rename = "source.discontinuity")]
    SourceDiscontinuity,
    #[serde(rename = "error")]
    Error,
}

impl CanonicalType {
    pub fn as_str(self) -> &'static str {
        match self {
            CanonicalType::SessionStarted => "session.started",
            CanonicalType::SessionResumed => "session.resumed",
            CanonicalType::SessionEnded => "session.ended",
            CanonicalType::UserMessage => "user.message",
            CanonicalType::AssistantDelta => "assistant.delta",
            CanonicalType::AssistantMessage => "assistant.message",
            CanonicalType::ToolCall => "tool.call",
            CanonicalType::ToolResult => "tool.result",
            CanonicalType::PermissionRequested => "permission.requested",
            CanonicalType::PermissionReplied => "permission.replied",
            CanonicalType::FileChanged => "file.changed",
            CanonicalType::CompactionBefore => "compaction.before",
            CanonicalType::CompactionAfter => "compaction.after",
            CanonicalType::SourceDiscontinuity => "source.discontinuity",
            CanonicalType::Error => "error",
        }
    }
}

impl FromStr for CanonicalType {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "session.started" => Ok(CanonicalType::SessionStarted),
            "session.resumed" => Ok(CanonicalType::SessionResumed),
            "session.ended" => Ok(CanonicalType::SessionEnded),
            "user.message" => Ok(CanonicalType::UserMessage),
            "assistant.delta" => Ok(CanonicalType::AssistantDelta),
            "assistant.message" => Ok(CanonicalType::AssistantMessage),
            "tool.call" => Ok(CanonicalType::ToolCall),
            "tool.result" => Ok(CanonicalType::ToolResult),
            "permission.requested" => Ok(CanonicalType::PermissionRequested),
            "permission.replied" => Ok(CanonicalType::PermissionReplied),
            "file.changed" => Ok(CanonicalType::FileChanged),
            "compaction.before" => Ok(CanonicalType::CompactionBefore),
            "compaction.after" => Ok(CanonicalType::CompactionAfter),
            "source.discontinuity" => Ok(CanonicalType::SourceDiscontinuity),
            "error" => Ok(CanonicalType::Error),
            _ => Err(Error::Validation(format!(
                "unsupported canonical_type: {value}"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventEnvelope {
    pub schema_version: u32,
    pub captured_at: String,
    pub tool: Tool,
    pub tool_version: Option<String>,
    pub session_id: String,
    pub filename_session_id: String,
    pub turn_id: Option<String>,
    pub message_id: Option<String>,
    pub project_root: Option<String>,
    pub cwd: Option<String>,
    pub source: Source,
    pub source_event_type: String,
    pub canonical_type: CanonicalType,
    pub source_event_id: Option<String>,
    pub dedupe_key: String,
    pub sequence: Option<i64>,
    pub raw_file: Option<String>,
    pub raw_offset: Option<i64>,
    pub payload: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload_ref: Option<String>,
}

impl EventEnvelope {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(Error::Validation(format!(
                "schema_version must be {SCHEMA_VERSION}"
            )));
        }
        if self.session_id.is_empty() {
            return Err(Error::Validation(
                "session_id must not be empty".to_string(),
            ));
        }
        if self.filename_session_id != sanitize_session_id(&self.session_id) {
            return Err(Error::Validation(
                "filename_session_id must match sanitized session_id".to_string(),
            ));
        }
        if self.source_event_type.is_empty() {
            return Err(Error::Validation(
                "source_event_type must not be empty".to_string(),
            ));
        }
        if !self.dedupe_key.starts_with("sha256:") {
            return Err(Error::Validation(
                "dedupe_key must start with sha256:".to_string(),
            ));
        }
        Ok(())
    }
}

pub struct DedupeParts<'a> {
    pub tool: Tool,
    pub session_id: &'a str,
    pub canonical_type: CanonicalType,
    pub source_event_id: Option<&'a str>,
    pub sequence: Option<i64>,
    pub payload: &'a Value,
}

pub fn dedupe_key(parts: DedupeParts<'_>) -> Result<String> {
    let mut hasher = Sha256::new();
    // Internal hash domain separator — intentionally NOT renamed with the product.
    // Changing this string re-keys every event and would duplicate/orphan existing
    // stores on reindex. Bump the version only on a deliberate identity change.
    hasher.update(b"harness-raven-dedupe-v2\0");
    hash_part(&mut hasher, parts.tool.as_str());
    hash_part(&mut hasher, parts.session_id);
    hash_part(&mut hasher, parts.canonical_type.as_str());

    if let Some(source_event_id) = parts.source_event_id {
        hash_part(&mut hasher, "native-id");
        hash_part(&mut hasher, source_event_id);
    } else {
        hash_part(&mut hasher, "content");
        hash_part(
            &mut hasher,
            &identity_content_hash(parts.canonical_type, parts.payload)?,
        );
        if let Some(sequence) = parts.sequence {
            hash_part(&mut hasher, "sequence");
            hash_part(&mut hasher, &sequence.to_string());
        } else {
            hash_part(&mut hasher, "unsequenced");
        }
    }

    Ok(format!("sha256:{}", hex::encode(hasher.finalize())))
}

fn hash_part(hasher: &mut Sha256, value: &str) {
    hasher.update(value.as_bytes());
    hasher.update([0]);
}

pub fn sanitize_session_id(session_id: &str) -> String {
    let sanitized: String = session_id
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-') {
                ch
            } else {
                '_'
            }
        })
        .collect();

    if sanitized.is_empty() {
        "_".to_string()
    } else {
        sanitized
    }
}

pub fn canonical_raw_path(home: &Path, tool: Tool, session_id: &str) -> PathBuf {
    let filename_session_id = sanitize_session_id(session_id);
    home.join("raw").join(tool.as_str()).join(format!(
        "{}_{}.jsonl",
        tool.as_str(),
        filename_session_id
    ))
}

pub fn resolve_home(cli_home: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(home) = cli_home {
        return Ok(home);
    }
    if let Some(home) = env::var_os("NABU_HOME") {
        return Ok(PathBuf::from(home));
    }
    // Deprecated pre-rename env var; accepted so existing setups keep working.
    if let Some(home) = env::var_os("TUPSHARRUM_HOME") {
        return Ok(PathBuf::from(home));
    }
    default_home()
}

pub fn default_home() -> Result<PathBuf> {
    let base = env::var_os("HOME")
        .map(PathBuf::from)
        .ok_or(Error::HomeUnavailable)?;
    let current = base.join(".nabu");
    // Back-compat: if the store has not been migrated yet, keep using the
    // pre-rename location so existing captured history is not orphaned.
    let legacy = base.join(".tupsharrum");
    if !current.exists() && legacy.exists() {
        return Ok(legacy);
    }
    Ok(current)
}

pub fn opencode_server_url(home: &Path) -> Result<Option<String>> {
    for key in ["NABU_OPENCODE_URL", "TUPSHARRUM_OPENCODE_URL"] {
        if let Some(value) = env::var_os(key) {
            let value = value.to_string_lossy().trim().to_string();
            if !value.is_empty() {
                return Ok(Some(value));
            }
        }
    }
    read_opencode_server_url_from_config(&home.join("config.toml"))
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

pub trait Embedder {
    fn embed_documents(&self, documents: &[String]) -> Result<Vec<Vec<f32>>>;
    fn embed_query(&self, query: &str) -> Result<Vec<f32>>;
    fn document_batch_size(&self) -> usize {
        16
    }
    fn intra_threads(&self) -> usize {
        1
    }
}

#[cfg(feature = "semantic")]
struct FastembedEmbedder {
    model: std::sync::Mutex<fastembed::TextEmbedding>,
    batch_size: usize,
    intra_threads: usize,
}

#[cfg(feature = "semantic")]
impl Embedder for FastembedEmbedder {
    fn embed_documents(&self, documents: &[String]) -> Result<Vec<Vec<f32>>> {
        let prompted = documents
            .iter()
            .map(|document| document_embedding_input(document))
            .collect::<Vec<_>>();
        let mut model = self.model.lock().map_err(|_| {
            Error::SemanticUnavailable("embedding model lock is poisoned".to_string())
        })?;
        let vectors = model
            .embed(&prompted, Some(self.batch_size))
            .map_err(|source| {
                Error::SemanticUnavailable(format!("document embedding failed: {source}"))
            })?;
        vectors
            .into_iter()
            .map(truncate_and_normalize_embedding)
            .collect()
    }

    fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        let prompted = [query_embedding_input(query)];
        let mut model = self.model.lock().map_err(|_| {
            Error::SemanticUnavailable("embedding model lock is poisoned".to_string())
        })?;
        let mut vectors = model.embed(&prompted, Some(1)).map_err(|source| {
            Error::SemanticUnavailable(format!("query embedding failed: {source}"))
        })?;
        let vector = vectors.pop().ok_or_else(|| {
            Error::SemanticUnavailable("query embedding returned no vector".to_string())
        })?;
        truncate_and_normalize_embedding(vector)
    }

    fn document_batch_size(&self) -> usize {
        self.batch_size
    }

    fn intra_threads(&self) -> usize {
        self.intra_threads
    }
}

#[cfg(any(feature = "semantic", test))]
fn document_embedding_input(document: &str) -> String {
    format!("{EMBEDDING_GEMMA_DOCUMENT_PREFIX}{}", document.trim())
}

#[cfg(any(feature = "semantic", test))]
fn query_embedding_input(query: &str) -> String {
    format!("{EMBEDDING_GEMMA_QUERY_PREFIX}{}", query.trim())
}

#[cfg(feature = "semantic")]
fn truncate_and_normalize_embedding(mut vector: Vec<f32>) -> Result<Vec<f32>> {
    if vector.len() < SEMANTIC_VECTOR_DIMENSIONS {
        return Err(Error::SemanticUnavailable(format!(
            "embedding returned {} dimensions, expected at least {}",
            vector.len(),
            SEMANTIC_VECTOR_DIMENSIONS
        )));
    }
    vector.truncate(SEMANTIC_VECTOR_DIMENSIONS);
    let norm = vector
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        .sqrt();
    if norm > 0.0 {
        for value in &mut vector {
            *value = (f64::from(*value) / norm) as f32;
        }
    }
    Ok(vector)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingUnitKind {
    UserText,
    AssistantText,
    ToolIntent,
    MetadataText,
}

impl EmbeddingUnitKind {
    pub fn as_str(self) -> &'static str {
        match self {
            EmbeddingUnitKind::UserText => "user_text",
            EmbeddingUnitKind::AssistantText => "assistant_text",
            EmbeddingUnitKind::ToolIntent => "tool_intent",
            EmbeddingUnitKind::MetadataText => "metadata_text",
        }
    }
}

impl FromStr for EmbeddingUnitKind {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "user_text" => Ok(Self::UserText),
            "assistant_text" => Ok(Self::AssistantText),
            "tool_intent" => Ok(Self::ToolIntent),
            "metadata_text" => Ok(Self::MetadataText),
            _ => Err(Error::Validation(format!(
                "unsupported embedding unit kind: {value}"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingUnit {
    pub kind: EmbeddingUnitKind,
    pub unit_index: usize,
    pub text: String,
    pub text_hash: String,
}

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

/// The complete, closed set of top-level entries nabu creates under a home.
/// A full purge only ever touches these; anything else is foreign and untouched.
const PURGE_KNOWN_ENTRIES: [&str; 9] = [
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FileSignature {
    len: u64,
    modified_nanos: u128,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SemanticStatusSignature {
    model_files: Option<Vec<FileSignature>>,
    index_files: Vec<Option<FileSignature>>,
}

#[derive(Debug, Clone)]
struct SemanticStatusCacheEntry {
    signature: SemanticStatusSignature,
    status: EmbeddingModelStatus,
}

#[cfg(feature = "semantic")]
struct CachedLocalEmbedder {
    model_files: Vec<FileSignature>,
    embedder: Arc<FastembedEmbedder>,
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct SourceCheckpoint {
    source_tool: Tool,
    source_kind: String,
    source_path: String,
    source_identity: Option<String>,
    session_id: String,
    byte_offset: u64,
    source_size: u64,
    source_mtime: Option<i64>,
    last_line_hash: Option<String>,
    last_successful_import_timestamp: String,
}

pub fn init_home(home: &Path) -> Result<InitReport> {
    let dirs = [
        home.to_path_buf(),
        home.join("raw"),
        home.join("raw").join("codex"),
        home.join("raw").join("claude"),
        home.join("raw").join("opencode"),
        home.join("spool"),
        home.join("spool").join("dedupe"),
        home.join("index"),
        home.join("models"),
        home.join("checkpoints"),
        home.join("blobs"),
        home.join("blobs").join("sha256"),
        home.join("logs"),
        home.join("backups"),
    ];

    for dir in dirs {
        create_dir_0700(&dir)?;
    }

    let config_path = home.join("config.toml");
    create_config_if_missing(&config_path)?;

    let db_path = home.join("index").join("harness.db");
    initialize_database(&db_path)?;

    Ok(InitReport {
        home: home.to_path_buf(),
        db_path,
    })
}

pub fn ingest_hook_event(home: &Path, tool: Tool, payload: Value) -> Result<AppendReport> {
    let source_event_type = hook_event_name(&payload)?.to_string();
    // OpenCode plugin events do not carry a top-level `session_id`; resolve from
    // the tool's own event shapes. Claude/Codex hooks emit `session_id` directly.
    let session_id = match tool {
        Tool::Opencode => opencode_hook_session_id(&payload, &source_event_type)?,
        _ => required_string(&payload, "session_id")?.to_string(),
    };
    let filename_session_id = sanitize_session_id(&session_id);
    let canonical_type = canonical_type_for_payload(tool, &source_event_type, &payload);
    let sequence = sequence_for_payload(tool, &source_event_type, &payload, None);
    let source_event_id = source_event_id_for_payload(tool, &source_event_type, &payload, sequence);
    let raw_file = canonical_raw_path(home, tool, &session_id);

    if let Some(parent) = raw_file.parent() {
        create_dir_0700(parent)?;
    }

    let lock_path = lock_path_for_raw_file(&raw_file);
    let lock_file = OpenOptions::new()
        .create(true)
        // Lock sentinel: content is never written, so do not truncate.
        .truncate(false)
        .read(true)
        .write(true)
        .open(&lock_path)
        .map_err(|source| Error::Io {
            path: lock_path.clone(),
            source,
        })?;
    chmod(&lock_path, 0o600)?;
    lock_file.lock_exclusive().map_err(|source| Error::Io {
        path: lock_path.clone(),
        source,
    })?;

    let append_result = append_envelope_locked(
        home,
        &raw_file,
        EventEnvelope {
            schema_version: SCHEMA_VERSION,
            captured_at: OffsetDateTime::now_utc().format(&Rfc3339)?,
            tool,
            tool_version: payload
                .get("tool_version")
                .and_then(Value::as_str)
                .map(str::to_string),
            session_id,
            filename_session_id,
            turn_id: payload
                .get("turn_id")
                .and_then(Value::as_str)
                .map(str::to_string),
            message_id: payload
                .get("message_id")
                .and_then(Value::as_str)
                .map(str::to_string),
            project_root: payload
                .get("project_root")
                .and_then(Value::as_str)
                .map(str::to_string),
            cwd: payload
                .get("cwd")
                .and_then(Value::as_str)
                .map(str::to_string),
            source: Source::Hook,
            source_event_type,
            canonical_type,
            source_event_id,
            dedupe_key: String::new(),
            sequence,
            raw_file: None,
            raw_offset: None,
            payload,
            payload_ref: None,
        },
    );

    let unlock_result = FileExt::unlock(&lock_file).map_err(|source| Error::Io {
        path: lock_path,
        source,
    });

    match (append_result, unlock_result) {
        (Ok(report), Ok(())) => Ok(report),
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error),
    }
}

pub fn ingest_file(
    home: &Path,
    tool: Tool,
    source: Source,
    path: &Path,
) -> Result<FileIngestReport> {
    let parsed = parse_ingest_file_source(tool, source, path)?;
    let mut events = parsed.events;
    for event in &mut events {
        event.source = source;
    }
    let appended_events = append_prepared_events(home, events)?
        .into_iter()
        .filter(|report| report.appended)
        .count();
    Ok(FileIngestReport { appended_events })
}

pub fn ingest_opencode_server_messages(
    home: &Path,
    session_id: &str,
    payload: Value,
) -> Result<FileIngestReport> {
    let events = opencode_server_events_from_payload(session_id, payload)?;
    let appended_events = append_prepared_events(home, events)?
        .into_iter()
        .filter(|report| report.appended)
        .count();
    Ok(FileIngestReport { appended_events })
}

fn append_envelope_locked(
    home: &Path,
    raw_file: &Path,
    envelope: EventEnvelope,
) -> Result<AppendReport> {
    let mut reports = append_envelopes_locked(home, raw_file, vec![envelope])?;
    reports
        .pop()
        .ok_or_else(|| Error::Validation("append produced no report".to_string()))
}

fn append_envelopes_locked(
    home: &Path,
    raw_file: &Path,
    events: Vec<EventEnvelope>,
) -> Result<Vec<AppendReport>> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .read(true)
        .open(raw_file)
        .map_err(|source| Error::Io {
            path: raw_file.to_path_buf(),
            source,
        })?;
    chmod(raw_file, 0o600)?;

    let mut keyed_events = Vec::with_capacity(events.len());
    let mut lookup_keys = HashSet::with_capacity(events.len());
    for mut envelope in events {
        envelope.dedupe_key = dedupe_key(DedupeParts {
            tool: envelope.tool,
            session_id: &envelope.session_id,
            canonical_type: envelope.canonical_type,
            source_event_id: envelope.source_event_id.as_deref(),
            sequence: envelope.sequence,
            payload: &envelope.payload,
        })?;
        lookup_keys.insert(envelope.dedupe_key.clone());
        keyed_events.push(envelope);
    }

    let mut dedupe_state = append_dedupe_state(home, raw_file, &lookup_keys)?;
    let mut raw_offset = file
        .metadata()
        .map_err(|source| Error::Io {
            path: raw_file.to_path_buf(),
            source,
        })?
        .len();
    let mut reports = Vec::with_capacity(keyed_events.len());

    for mut envelope in keyed_events {
        if let Some(existing) = dedupe_state.existing(&envelope.dedupe_key) {
            reports.push(AppendReport {
                raw_file: raw_file.to_path_buf(),
                raw_offset: existing.raw_offset,
                session_id: envelope.session_id,
                dedupe_key: envelope.dedupe_key,
                appended: false,
            });
            continue;
        }

        let event_raw_offset = raw_offset;
        envelope.raw_file = Some(raw_file.display().to_string());
        envelope.raw_offset = Some(event_raw_offset as i64);
        spill_payload_if_needed(home, &mut envelope)?;
        envelope.validate()?;

        let line = serde_json::to_vec(&envelope)?;
        file.write_all(&line).map_err(|source| Error::Io {
            path: raw_file.to_path_buf(),
            source,
        })?;
        file.write_all(b"\n").map_err(|source| Error::Io {
            path: raw_file.to_path_buf(),
            source,
        })?;

        let line_len = line.len() as u64 + 1;
        dedupe_state.record_appended(
            envelope.dedupe_key.clone(),
            ExistingRawEvent {
                raw_offset: event_raw_offset,
            },
            line_len,
        );
        raw_offset += line_len;

        reports.push(AppendReport {
            raw_file: raw_file.to_path_buf(),
            raw_offset: event_raw_offset,
            session_id: envelope.session_id,
            dedupe_key: envelope.dedupe_key,
            appended: true,
        });
    }

    dedupe_state.flush();

    Ok(reports)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ExistingRawEvent {
    raw_offset: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RawDedupeSnapshot {
    events: HashMap<String, ExistingRawEvent>,
    ordered: Vec<(String, u64)>,
    raw_len: u64,
}

impl RawDedupeSnapshot {
    fn empty(raw_len: u64) -> Self {
        Self {
            events: HashMap::new(),
            ordered: Vec::new(),
            raw_len,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AppendDedupeState {
    events: HashMap<String, ExistingRawEvent>,
    pending: Vec<(String, u64)>,
    raw_len: u64,
    key_count: usize,
    bucket_lengths: Vec<u64>,
    sidecar: Option<DedupeSidecarFiles>,
}

impl AppendDedupeState {
    fn from_sidecar(
        events: HashMap<String, ExistingRawEvent>,
        raw_len: u64,
        key_count: usize,
        bucket_lengths: Vec<u64>,
        sidecar: DedupeSidecarFiles,
    ) -> Self {
        Self {
            events,
            pending: Vec::new(),
            raw_len,
            key_count,
            bucket_lengths,
            sidecar: Some(sidecar),
        }
    }

    fn from_snapshot(
        snapshot: RawDedupeSnapshot,
        sidecar: Option<DedupeSidecarFiles>,
        bucket_lengths: Vec<u64>,
        lookup_keys: &HashSet<String>,
    ) -> Self {
        let key_count = snapshot.ordered.len();
        let events = snapshot
            .events
            .into_iter()
            .filter(|(dedupe_key, _)| lookup_keys.contains(dedupe_key))
            .collect();
        Self {
            events,
            pending: Vec::new(),
            raw_len: snapshot.raw_len,
            key_count,
            bucket_lengths,
            sidecar,
        }
    }

    fn existing(&self, dedupe_key: &str) -> Option<&ExistingRawEvent> {
        self.events.get(dedupe_key)
    }

    fn record_appended(
        &mut self,
        dedupe_key: String,
        existing: ExistingRawEvent,
        raw_line_len: u64,
    ) {
        self.pending.push((dedupe_key.clone(), existing.raw_offset));
        self.events.entry(dedupe_key).or_insert(existing);
        self.raw_len = self.raw_len.saturating_add(raw_line_len);
        self.key_count = self.key_count.saturating_add(1);
    }

    fn flush(&mut self) {
        let Some(sidecar) = self.sidecar.as_ref() else {
            return;
        };
        if self.pending.is_empty() {
            return;
        }
        match append_dedupe_sidecar(sidecar, self) {
            Ok(bucket_lengths) => {
                self.bucket_lengths = bucket_lengths;
                if let Err(error) = write_dedupe_sidecar_meta(
                    sidecar,
                    self.raw_len,
                    self.key_count,
                    &self.bucket_lengths,
                ) {
                    eprintln!(
                        "nabu: dedupe sidecar metadata update failed at {}: {}; future appends will rebuild or fall back to raw",
                        sidecar.meta.display(),
                        error
                    );
                    self.sidecar = None;
                    return;
                }
            }
            Err(error) => {
                eprintln!(
                    "nabu: dedupe sidecar update failed at {}: {}; future appends will rebuild or fall back to raw",
                    sidecar.meta.display(),
                    error
                );
                self.sidecar = None;
                return;
            }
        }
        self.pending.clear();
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DedupeSidecarFiles {
    buckets_dir: PathBuf,
    meta: PathBuf,
    legacy_keys: PathBuf,
    legacy_offsets: PathBuf,
}

impl DedupeSidecarFiles {
    fn for_raw_file(home: &Path, raw_file: &Path) -> Self {
        let base = raw_file
            .file_stem()
            .and_then(|value| value.to_str())
            .filter(|value| !value.is_empty())
            .unwrap_or("raw");
        let dir = home.join("spool").join("dedupe");
        Self {
            buckets_dir: dir.join(format!("{base}.buckets")),
            meta: dir.join(format!("{base}.meta.json")),
            legacy_keys: dir.join(format!("{base}.keys")),
            legacy_offsets: dir.join(format!("{base}.offsets")),
        }
    }

    fn bucket_path(&self, bucket: usize) -> PathBuf {
        self.buckets_dir.join(format!("{bucket:02x}.dedupe"))
    }

    fn file_paths(&self) -> [&Path; 3] {
        [&self.meta, &self.legacy_keys, &self.legacy_offsets]
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct DedupeSidecarMeta {
    schema_version: u32,
    raw_len: u64,
    key_count: usize,
    bucket_count: usize,
    bucket_lengths: Vec<u64>,
}

const DEDUPE_SIDECAR_SCHEMA_VERSION: u32 = 2;
const DEDUPE_BUCKET_COUNT: usize = 256;

fn append_dedupe_state(
    home: &Path,
    raw_file: &Path,
    lookup_keys: &HashSet<String>,
) -> Result<AppendDedupeState> {
    let sidecar = DedupeSidecarFiles::for_raw_file(home, raw_file);
    match load_append_dedupe_sidecar(raw_file, &sidecar, lookup_keys) {
        Ok(Some(state)) => Ok(state),
        Ok(None) => rebuild_dedupe_state(raw_file, sidecar, lookup_keys),
        Err(error) => {
            eprintln!(
                "nabu: dedupe sidecar read failed at {}: {}; falling back to raw-derived check",
                sidecar.meta.display(),
                error
            );
            Ok(AppendDedupeState::from_snapshot(
                read_raw_dedupe_snapshot(raw_file)?,
                None,
                zero_bucket_lengths(),
                lookup_keys,
            ))
        }
    }
}

fn rebuild_dedupe_state(
    raw_file: &Path,
    sidecar: DedupeSidecarFiles,
    lookup_keys: &HashSet<String>,
) -> Result<AppendDedupeState> {
    let snapshot = read_raw_dedupe_snapshot(raw_file)?;
    match write_full_dedupe_sidecar(&sidecar, &snapshot) {
        Ok(bucket_lengths) => Ok(AppendDedupeState::from_snapshot(
            snapshot,
            Some(sidecar),
            bucket_lengths,
            lookup_keys,
        )),
        Err(error) => {
            eprintln!(
                "nabu: dedupe sidecar rebuild failed at {}: {}; falling back to raw-derived check",
                sidecar.meta.display(),
                error
            );
            Ok(AppendDedupeState::from_snapshot(
                snapshot,
                None,
                zero_bucket_lengths(),
                lookup_keys,
            ))
        }
    }
}

fn load_append_dedupe_sidecar(
    raw_file: &Path,
    sidecar: &DedupeSidecarFiles,
    lookup_keys: &HashSet<String>,
) -> Result<Option<AppendDedupeState>> {
    let Some((meta, raw_len)) = read_dedupe_sidecar_meta(raw_file, sidecar)? else {
        return Ok(None);
    };
    let mut events = HashMap::new();
    let mut buckets = BTreeMap::<usize, HashSet<String>>::new();
    for dedupe_key in lookup_keys {
        let Some(bucket) = dedupe_bucket_index(dedupe_key) else {
            return Ok(None);
        };
        buckets
            .entry(bucket)
            .or_default()
            .insert(dedupe_key.clone());
    }

    for (bucket, needed) in buckets {
        if !load_dedupe_bucket(
            sidecar,
            bucket,
            meta.bucket_lengths[bucket],
            Some(&needed),
            &mut events,
        )? {
            return Ok(None);
        }
    }

    Ok(Some(AppendDedupeState::from_sidecar(
        events,
        raw_len,
        meta.key_count,
        meta.bucket_lengths,
        sidecar.clone(),
    )))
}

fn load_full_dedupe_sidecar_events(
    raw_file: &Path,
    sidecar: &DedupeSidecarFiles,
) -> Result<Option<HashMap<String, ExistingRawEvent>>> {
    let Some((meta, _)) = read_dedupe_sidecar_meta(raw_file, sidecar)? else {
        return Ok(None);
    };
    let mut events = HashMap::new();
    for bucket in 0..DEDUPE_BUCKET_COUNT {
        if !load_dedupe_bucket(
            sidecar,
            bucket,
            meta.bucket_lengths[bucket],
            None,
            &mut events,
        )? {
            return Ok(None);
        }
    }
    if events.len() > meta.key_count {
        return Ok(None);
    }
    Ok(Some(events))
}

fn read_dedupe_sidecar_meta(
    raw_file: &Path,
    sidecar: &DedupeSidecarFiles,
) -> Result<Option<(DedupeSidecarMeta, u64)>> {
    let raw_len = match fs::metadata(raw_file) {
        Ok(metadata) => metadata.len(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0,
        Err(source) => {
            return Err(Error::Io {
                path: raw_file.to_path_buf(),
                source,
            })
        }
    };
    let meta_bytes = match fs::read(&sidecar.meta) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(source) => {
            return Err(Error::Io {
                path: sidecar.meta.clone(),
                source,
            })
        }
    };
    let Ok(meta) = serde_json::from_slice::<DedupeSidecarMeta>(&meta_bytes) else {
        return Ok(None);
    };
    if meta.schema_version != DEDUPE_SIDECAR_SCHEMA_VERSION
        || meta.raw_len != raw_len
        || meta.bucket_count != DEDUPE_BUCKET_COUNT
        || meta.bucket_lengths.len() != DEDUPE_BUCKET_COUNT
    {
        return Ok(None);
    }

    Ok(Some((meta, raw_len)))
}

fn load_dedupe_bucket(
    sidecar: &DedupeSidecarFiles,
    bucket: usize,
    expected_len: u64,
    needed: Option<&HashSet<String>>,
    events: &mut HashMap<String, ExistingRawEvent>,
) -> Result<bool> {
    let path = sidecar.bucket_path(bucket);
    match fs::metadata(&path) {
        Ok(metadata) if metadata.len() == expected_len => {}
        Ok(_) => return Ok(false),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound && expected_len == 0 => {
            return Ok(true)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(source) => return Err(Error::Io { path, source }),
    }
    let file = match File::open(&path) {
        Ok(file) => file,
        Err(source) => return Err(Error::Io { path, source }),
    };
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    loop {
        line.clear();
        let bytes = reader.read_line(&mut line).map_err(|source| Error::Io {
            path: path.clone(),
            source,
        })?;
        if bytes == 0 {
            break;
        }
        let Some((dedupe_key, raw_offset)) = parse_dedupe_bucket_entry(line.trim_end()) else {
            return Ok(false);
        };
        if dedupe_bucket_index(dedupe_key) != Some(bucket) {
            return Ok(false);
        }
        if needed.map(|keys| keys.contains(dedupe_key)).unwrap_or(true) {
            events
                .entry(dedupe_key.to_string())
                .or_insert(ExistingRawEvent { raw_offset });
        }
    }
    Ok(true)
}

fn parse_dedupe_bucket_entry(line: &str) -> Option<(&str, u64)> {
    let (dedupe_key, raw_offset) = line.split_once('\t')?;
    if !valid_dedupe_key(dedupe_key) {
        return None;
    }
    Some((dedupe_key, raw_offset.parse::<u64>().ok()?))
}

fn valid_dedupe_key(value: &str) -> bool {
    value.len() == "sha256:".len() + 64
        && value.starts_with("sha256:")
        && value["sha256:".len()..]
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
}

fn read_raw_dedupe_snapshot(raw_file: &Path) -> Result<RawDedupeSnapshot> {
    if !raw_file.exists() {
        return Ok(RawDedupeSnapshot::empty(0));
    }

    let file = File::open(raw_file).map_err(|source| Error::Io {
        path: raw_file.to_path_buf(),
        source,
    })?;
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    let mut raw_offset = 0u64;
    let mut events = HashMap::new();
    let mut ordered = Vec::new();

    loop {
        line.clear();
        let bytes = reader.read_line(&mut line).map_err(|source| Error::Io {
            path: raw_file.to_path_buf(),
            source,
        })?;
        if bytes == 0 {
            break;
        }
        let parsed: EventEnvelope = serde_json::from_str(line.trim_end())?;
        if !parsed.dedupe_key.is_empty() {
            let event_raw_offset = parsed.raw_offset.unwrap_or(raw_offset as i64).max(0) as u64;
            ordered.push((parsed.dedupe_key.clone(), event_raw_offset));
            events.entry(parsed.dedupe_key).or_insert(ExistingRawEvent {
                raw_offset: event_raw_offset,
            });
        }
        raw_offset += bytes as u64;
    }

    Ok(RawDedupeSnapshot {
        events,
        ordered,
        raw_len: raw_offset,
    })
}

fn write_full_dedupe_sidecar(
    sidecar: &DedupeSidecarFiles,
    snapshot: &RawDedupeSnapshot,
) -> Result<Vec<u64>> {
    let Some(parent) = sidecar.meta.parent() else {
        return Err(Error::Validation(
            "dedupe sidecar has no parent".to_string(),
        ));
    };
    create_dir_0700(parent)?;
    match fs::remove_dir_all(&sidecar.buckets_dir) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(source) => {
            return Err(Error::Io {
                path: sidecar.buckets_dir.clone(),
                source,
            })
        }
    }
    create_dir_0700(&sidecar.buckets_dir)?;

    let mut bucket_lengths = zero_bucket_lengths();
    let mut bucket_files = (0..DEDUPE_BUCKET_COUNT)
        .map(|_| None)
        .collect::<Vec<Option<File>>>();
    for (dedupe_key, raw_offset) in &snapshot.ordered {
        let Some(bucket) = dedupe_bucket_index(dedupe_key) else {
            return Err(Error::Validation(format!(
                "invalid dedupe key in raw snapshot: {dedupe_key}"
            )));
        };
        if bucket_files[bucket].is_none() {
            let path = sidecar.bucket_path(bucket);
            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .map_err(|source| Error::Io {
                    path: path.clone(),
                    source,
                })?;
            chmod(&path, 0o600)?;
            bucket_files[bucket] = Some(file);
        }
        let entry = dedupe_bucket_entry(dedupe_key, *raw_offset);
        if let Some(file) = bucket_files[bucket].as_mut() {
            file.write_all(entry.as_bytes())
                .map_err(|source| Error::Io {
                    path: sidecar.bucket_path(bucket),
                    source,
                })?;
        }
        bucket_lengths[bucket] = bucket_lengths[bucket].saturating_add(entry.len() as u64);
    }
    drop(bucket_files);

    write_dedupe_sidecar_meta(
        sidecar,
        snapshot.raw_len,
        snapshot.ordered.len(),
        &bucket_lengths,
    )?;
    Ok(bucket_lengths)
}

fn append_dedupe_sidecar(
    sidecar: &DedupeSidecarFiles,
    state: &AppendDedupeState,
) -> Result<Vec<u64>> {
    let Some(parent) = sidecar.meta.parent() else {
        return Err(Error::Validation(
            "dedupe sidecar has no parent".to_string(),
        ));
    };
    create_dir_0700(parent)?;
    create_dir_0700(&sidecar.buckets_dir)?;

    let mut bucket_lengths = state.bucket_lengths.clone();
    if bucket_lengths.len() != DEDUPE_BUCKET_COUNT {
        return Err(Error::Validation(
            "dedupe sidecar bucket metadata is invalid".to_string(),
        ));
    }
    let mut pending_by_bucket = BTreeMap::<usize, Vec<&(String, u64)>>::new();
    for entry in &state.pending {
        let Some(bucket) = dedupe_bucket_index(&entry.0) else {
            return Err(Error::Validation(format!(
                "invalid pending dedupe key: {}",
                entry.0
            )));
        };
        pending_by_bucket.entry(bucket).or_default().push(entry);
    }

    for (bucket, entries) in pending_by_bucket {
        let path = sidecar.bucket_path(bucket);
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|source| Error::Io {
                path: path.clone(),
                source,
            })?;
        for (dedupe_key, raw_offset) in entries {
            let entry = dedupe_bucket_entry(dedupe_key, *raw_offset);
            file.write_all(entry.as_bytes())
                .map_err(|source| Error::Io {
                    path: path.clone(),
                    source,
                })?;
            bucket_lengths[bucket] = bucket_lengths[bucket].saturating_add(entry.len() as u64);
        }
        chmod(&path, 0o600)?;
    }
    Ok(bucket_lengths)
}

fn write_dedupe_sidecar_meta(
    sidecar: &DedupeSidecarFiles,
    raw_len: u64,
    key_count: usize,
    bucket_lengths: &[u64],
) -> Result<()> {
    let meta = DedupeSidecarMeta {
        schema_version: DEDUPE_SIDECAR_SCHEMA_VERSION,
        raw_len,
        key_count,
        bucket_count: DEDUPE_BUCKET_COUNT,
        bucket_lengths: bucket_lengths.to_vec(),
    };
    let bytes = serde_json::to_vec_pretty(&meta)?;
    fs::write(&sidecar.meta, bytes).map_err(|source| Error::Io {
        path: sidecar.meta.clone(),
        source,
    })?;
    chmod(&sidecar.meta, 0o600)
}

fn dedupe_bucket_index(dedupe_key: &str) -> Option<usize> {
    if !valid_dedupe_key(dedupe_key) {
        return None;
    }
    usize::from_str_radix(&dedupe_key["sha256:".len().."sha256:".len() + 2], 16).ok()
}

fn dedupe_bucket_entry(dedupe_key: &str, raw_offset: u64) -> String {
    format!("{dedupe_key}\t{raw_offset}\n")
}

fn zero_bucket_lengths() -> Vec<u64> {
    vec![0; DEDUPE_BUCKET_COUNT]
}

fn remove_dedupe_sidecar_for_raw_file(raw_file: &Path) -> Result<()> {
    let home = harness_home_for_raw_file(raw_file);
    let sidecar = DedupeSidecarFiles::for_raw_file(&home, raw_file);
    for path in sidecar.file_paths() {
        match fs::remove_file(path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(source) => {
                return Err(Error::Io {
                    path: path.to_path_buf(),
                    source,
                })
            }
        }
    }
    match fs::remove_dir_all(&sidecar.buckets_dir) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(source) => {
            return Err(Error::Io {
                path: sidecar.buckets_dir,
                source,
            })
        }
    }
    Ok(())
}

fn spill_payload_if_needed(home: &Path, envelope: &mut EventEnvelope) -> Result<()> {
    if serde_json::to_vec(envelope)?.len() <= MAX_INLINE_ENVELOPE_BYTES {
        return Ok(());
    }

    let payload_bytes = serde_json::to_vec(&envelope.payload)?;
    let mut hasher = Sha256::new();
    hasher.update(&payload_bytes);
    let hash = hex::encode(hasher.finalize());
    let blob_dir = home.join("blobs").join("sha256");
    create_dir_0700(&blob_dir)?;
    let blob_path = blob_dir.join(format!("{hash}.json"));
    if !blob_path.exists() {
        fs::write(&blob_path, &payload_bytes).map_err(|source| Error::Io {
            path: blob_path.clone(),
            source,
        })?;
        chmod(&blob_path, 0o600)?;
    }
    envelope.payload = Value::Null;
    envelope.payload_ref = Some(format!("sha256:{hash}"));
    Ok(())
}

pub fn index_once(home: &Path) -> Result<IndexReport> {
    index_once_with_progress(home, |_| {})
}

pub fn index_once_with_progress<F>(home: &Path, progress: F) -> Result<IndexReport>
where
    F: FnMut(EmbeddingIndexProgress),
{
    index_once_with_options_and_progress(home, IndexOptions::default(), progress)
}

pub fn index_once_with_options(home: &Path, options: IndexOptions) -> Result<IndexReport> {
    index_once_with_options_and_progress(home, options, |_| {})
}

pub fn index_once_with_options_and_progress<F>(
    home: &Path,
    options: IndexOptions,
    progress: F,
) -> Result<IndexReport>
where
    F: FnMut(EmbeddingIndexProgress),
{
    init_home(home)?;
    let db_path = home.join("index").join("harness.db");
    let mut conn = open_index(&db_path)?;
    ensure_semantic_vector_schema(&conn, &db_path)?;
    let tx = conn.transaction().map_err(|source| Error::Sqlite {
        path: db_path.clone(),
        source,
    })?;

    let mut indexed_events = 0usize;
    for tool in Tool::all() {
        let raw_dir = home.join("raw").join(tool.as_str());
        if !raw_dir.exists() {
            continue;
        }

        let entries = fs::read_dir(&raw_dir).map_err(|source| Error::Io {
            path: raw_dir.clone(),
            source,
        })?;
        for entry in entries {
            let entry = entry.map_err(|source| Error::Io {
                path: raw_dir.clone(),
                source,
            })?;
            let path = entry.path();
            if path.extension().and_then(|value| value.to_str()) != Some("jsonl") {
                continue;
            }
            let source_meta = source_file_metadata(&path)?;
            if raw_index_checkpoint_is_current(&tx, &db_path, tool, &path, &source_meta)? {
                continue;
            }

            let raw_report = index_raw_file(&tx, tool, &path)?;
            indexed_events += raw_report.indexed_events;
            write_raw_index_checkpoint(&tx, &db_path, tool, &path, source_meta, raw_report)?;
        }
    }

    recalculate_session_counts(&tx)?;
    tx.commit().map_err(|source| Error::Sqlite {
        path: db_path.clone(),
        source,
    })?;

    if options.embed {
        embed_index_if_available_with_progress(home, progress)?;
    }

    Ok(IndexReport { indexed_events })
}

pub fn search_history(home: &Path, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
    search_history_filtered(
        home,
        query,
        SearchOptions {
            limit,
            ..SearchOptions::default()
        },
    )
}

pub fn search_history_filtered(
    home: &Path,
    query: &str,
    options: SearchOptions,
) -> Result<Vec<SearchResult>> {
    Ok(search_history_page(home, query, options)?.results)
}

pub fn embedding_model_status(home: &Path) -> EmbeddingModelStatus {
    let signature = semantic_status_signature(home);
    let cache_key = home.to_path_buf();
    if let Ok(cache) = semantic_status_cache().lock() {
        if let Some(entry) = cache.get(&cache_key) {
            if entry.signature == signature {
                return entry.status.clone();
            }
        }
    }

    let status = embedding_model_status_uncached(home, &signature);
    if let Ok(mut cache) = semantic_status_cache().lock() {
        cache.insert(
            cache_key,
            SemanticStatusCacheEntry {
                signature,
                status: status.clone(),
            },
        );
    }
    status
}

fn embedding_model_status_uncached(
    home: &Path,
    signature: &SemanticStatusSignature,
) -> EmbeddingModelStatus {
    let cache_path = semantic_model_cache_path(home);
    let feature_enabled = cfg!(feature = "semantic");
    let model_present = signature.model_files.is_some();
    let vector_rows = if feature_enabled && model_present {
        semantic_vector_row_count(home).unwrap_or(0)
    } else {
        0
    };
    let semantic_available = feature_enabled && model_present && vector_rows > 0;
    let message = if !feature_enabled {
        "semantic feature is disabled in this build".to_string()
    } else if !model_present {
        "semantic feature is enabled, but the local model is not installed".to_string()
    } else if vector_rows == 0 {
        "semantic feature is enabled and the local model is installed, but the vector index has no embeddings; run nabu index --once".to_string()
    } else {
        "semantic search is available".to_string()
    };

    EmbeddingModelStatus {
        feature_enabled,
        model_id: SEMANTIC_MODEL_ID.to_string(),
        model_present,
        semantic_available,
        cache_path: cache_path.display().to_string(),
        expected_dimensions: SEMANTIC_VECTOR_DIMENSIONS,
        message,
    }
}

fn semantic_status_cache() -> &'static Mutex<HashMap<PathBuf, SemanticStatusCacheEntry>> {
    static CACHE: OnceLock<Mutex<HashMap<PathBuf, SemanticStatusCacheEntry>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn prune_embedding_cache(home: &Path) -> Result<StorageFootprint> {
    let model_root = home.join("models");
    if model_root.exists() {
        fs::remove_dir_all(&model_root).map_err(|source| Error::Io {
            path: model_root.clone(),
            source,
        })?;
    }
    create_dir_0700(&model_root)?;
    Ok(storage_footprint(home))
}

pub fn embedding_model_disclosure(home: &Path, model: &str) -> Result<EmbeddingModelDisclosure> {
    if model != SEMANTIC_MODEL_ID {
        return Err(Error::Validation(format!(
            "unsupported embedding model: {model}"
        )));
    }
    let cache_path = semantic_model_cache_path(home);
    let current_on_disk_bytes = directory_size(&cache_path).unwrap_or(0);
    Ok(EmbeddingModelDisclosure {
        model_id: SEMANTIC_MODEL_ID.to_string(),
        repository: SEMANTIC_MODEL_REPO.to_string(),
        cache_path: cache_path.display().to_string(),
        total_files: SEMANTIC_MODEL_REMOTE_FILES.len(),
        current_on_disk_bytes,
        model_present: semantic_model_files_present(home),
        license_summary: gemma_terms_summary().to_string(),
    })
}

fn gemma_terms_summary() -> &'static str {
    "Gemma Terms of Use: open-weight license permitting responsible commercial use, fine-tuning, and redistribution; no per-token fees."
}

pub fn download_embedding_model(home: &Path, model: &str) -> Result<EmbeddingDownloadReport> {
    download_embedding_model_with_progress(home, model, |_| {})
}

#[cfg(feature = "semantic")]
pub fn download_embedding_model_with_progress<F>(
    home: &Path,
    model: &str,
    mut progress: F,
) -> Result<EmbeddingDownloadReport>
where
    F: FnMut(EmbeddingDownloadProgress),
{
    if model != SEMANTIC_MODEL_ID {
        return Err(Error::Validation(format!(
            "unsupported embedding model: {model}"
        )));
    }

    init_home(home)?;
    let cache_path = semantic_model_cache_path(home);
    create_dir_0700(&cache_path)?;
    let transient_cache = cache_path.join(".hf-download-cache");
    if transient_cache.exists() {
        fs::remove_dir_all(&transient_cache).map_err(|source| Error::Io {
            path: transient_cache.clone(),
            source,
        })?;
    }
    create_dir_0700(&transient_cache)?;

    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_cache_dir(transient_cache.clone())
        .with_progress(false)
        .build()
        .map_err(|source| Error::SemanticUnavailable(format!("model download failed: {source}")))?;
    let repo = api.model(SEMANTIC_MODEL_REPO.to_string());
    let total_files = SEMANTIC_MODEL_REMOTE_FILES.len();
    let mut downloaded_files = 0usize;
    let mut downloaded_bytes = 0u64;

    for (remote, local) in SEMANTIC_MODEL_REMOTE_FILES {
        progress(EmbeddingDownloadProgress {
            model_id: SEMANTIC_MODEL_ID.to_string(),
            file: (*remote).to_string(),
            downloaded_files,
            total_files,
            phase: "downloading".to_string(),
        });
        let source_path = repo.get(remote).map_err(|source| {
            Error::SemanticUnavailable(format!("model download failed for {remote}: {source}"))
        })?;
        let source_path = fs::canonicalize(&source_path).unwrap_or(source_path);
        let target_path = cache_path.join(local);
        if let Some(parent) = target_path.parent() {
            create_dir_0700(parent)?;
        }
        fs::copy(&source_path, &target_path).map_err(|source| Error::Io {
            path: target_path.clone(),
            source,
        })?;
        downloaded_bytes = downloaded_bytes.saturating_add(
            fs::metadata(&target_path)
                .map_err(|source| Error::Io {
                    path: target_path.clone(),
                    source,
                })?
                .len(),
        );
        chmod(&target_path, 0o600)?;
        downloaded_files += 1;
        progress(EmbeddingDownloadProgress {
            model_id: SEMANTIC_MODEL_ID.to_string(),
            file: (*remote).to_string(),
            downloaded_files,
            total_files,
            phase: "stored".to_string(),
        });
    }

    fs::remove_dir_all(&transient_cache).map_err(|source| Error::Io {
        path: transient_cache,
        source,
    })?;

    Ok(EmbeddingDownloadReport {
        model_id: SEMANTIC_MODEL_ID.to_string(),
        cache_path: cache_path.display().to_string(),
        downloaded_files,
        total_files,
        downloaded_bytes,
        on_disk_bytes: directory_size(&cache_path).unwrap_or(downloaded_bytes),
        license_summary: gemma_terms_summary().to_string(),
    })
}

#[cfg(not(feature = "semantic"))]
pub fn download_embedding_model_with_progress<F>(
    _home: &Path,
    _model: &str,
    _progress: F,
) -> Result<EmbeddingDownloadReport>
where
    F: FnMut(EmbeddingDownloadProgress),
{
    Err(Error::SemanticUnavailable(
        "semantic backend is not available in this build; rebuild with --features semantic to enable explicit model download".to_string(),
    ))
}

fn semantic_search_available(home: &Path) -> bool {
    if !cfg!(feature = "semantic") {
        return false;
    }
    embedding_model_status(home).semantic_available
}

fn semantic_model_cache_path(home: &Path) -> PathBuf {
    home.join("models").join(SEMANTIC_MODEL_ID)
}

fn semantic_model_files_present(home: &Path) -> bool {
    semantic_model_file_signatures(home).is_some()
}

fn semantic_status_signature(home: &Path) -> SemanticStatusSignature {
    let model_files = semantic_model_file_signatures(home);
    let index_files = if cfg!(feature = "semantic") && model_files.is_some() {
        semantic_index_file_signatures(home)
    } else {
        Vec::new()
    };
    SemanticStatusSignature {
        model_files,
        index_files,
    }
}

fn semantic_index_file_signatures(home: &Path) -> Vec<Option<FileSignature>> {
    let db_path = home.join("index").join("harness.db");
    vec![
        file_signature(&db_path),
        file_signature(&db_path.with_file_name("harness.db-wal")),
        file_signature(&db_path.with_file_name("harness.db-shm")),
    ]
}

fn semantic_model_file_signatures(home: &Path) -> Option<Vec<FileSignature>> {
    let cache_path = semantic_model_cache_path(home);
    let mut signatures = Vec::with_capacity(SEMANTIC_MODEL_REMOTE_FILES.len());
    for (_, local) in SEMANTIC_MODEL_REMOTE_FILES {
        let path = cache_path.join(local);
        if !path.is_file() {
            return None;
        }
        signatures.push(file_signature(&path)?);
    }
    Some(signatures)
}

fn file_signature(path: &Path) -> Option<FileSignature> {
    let metadata = fs::metadata(path).ok()?;
    let modified_nanos = metadata
        .modified()
        .ok()
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    Some(FileSignature {
        len: metadata.len(),
        modified_nanos,
    })
}

fn semantic_vector_row_count(home: &Path) -> Result<i64> {
    let db_path = home.join("index").join("harness.db");
    if !db_path.exists() {
        return Ok(0);
    }
    let conn = open_index(&db_path)?;
    if !table_exists(&conn, &db_path, "vector_unit_embeddings")? {
        return Ok(0);
    }
    table_count(&conn, &db_path, "vector_unit_embeddings")
}

#[cfg(feature = "semantic")]
fn semantic_intra_threads() -> usize {
    let available = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .max(1);
    let requested = env::var("NABU_SEMANTIC_INTRA_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .or_else(platform_physical_core_count)
        .unwrap_or(available);
    requested.clamp(1, available)
}

#[cfg(all(feature = "semantic", target_os = "macos"))]
fn platform_physical_core_count() -> Option<usize> {
    let mut value: libc::c_int = 0;
    let mut size = std::mem::size_of::<libc::c_int>();
    let status = unsafe {
        libc::sysctlbyname(
            b"hw.physicalcpu\0".as_ptr().cast(),
            (&mut value as *mut libc::c_int).cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    (status == 0 && value > 0).then_some(value as usize)
}

#[cfg(all(feature = "semantic", target_os = "linux"))]
fn platform_physical_core_count() -> Option<usize> {
    let cpuinfo = fs::read_to_string("/proc/cpuinfo").ok()?;
    parse_linux_physical_core_count(&cpuinfo)
}

#[cfg(all(
    feature = "semantic",
    not(any(target_os = "linux", target_os = "macos"))
))]
fn platform_physical_core_count() -> Option<usize> {
    None
}

#[cfg(all(feature = "semantic", target_os = "linux"))]
fn parse_linux_physical_core_count(cpuinfo: &str) -> Option<usize> {
    let mut physical_cores = HashSet::new();
    let mut processors = 0usize;
    let mut physical_id: Option<String> = None;
    let mut core_id: Option<String> = None;

    for line in cpuinfo.lines().chain(std::iter::once("")) {
        let line = line.trim();
        if line.is_empty() {
            if let (Some(package), Some(core)) = (physical_id.take(), core_id.take()) {
                physical_cores.insert((package, core));
            } else {
                physical_id = None;
                core_id = None;
            }
            continue;
        }
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        match key.trim() {
            "processor" => processors = processors.saturating_add(1),
            "physical id" => physical_id = Some(value.trim().to_string()),
            "core id" => core_id = Some(value.trim().to_string()),
            _ => {}
        }
    }

    if !physical_cores.is_empty() {
        Some(physical_cores.len())
    } else if processors > 0 {
        Some(processors)
    } else {
        None
    }
}

#[cfg(feature = "semantic")]
fn load_local_embedder(home: &Path) -> Result<Option<Arc<FastembedEmbedder>>> {
    let Some(model_files) = semantic_model_file_signatures(home) else {
        return Ok(None);
    };
    let cache_key = semantic_model_cache_path(home);
    if let Ok(cache) = local_embedder_cache().lock() {
        if let Some(entry) = cache.get(&cache_key) {
            if entry.model_files == model_files {
                return Ok(Some(Arc::clone(&entry.embedder)));
            }
        }
    }

    let embedder = Arc::new(load_local_embedder_uncached(home)?);
    if let Ok(mut cache) = local_embedder_cache().lock() {
        cache.insert(
            cache_key,
            CachedLocalEmbedder {
                model_files,
                embedder: Arc::clone(&embedder),
            },
        );
    }
    Ok(Some(embedder))
}

#[cfg(feature = "semantic")]
fn local_embedder_cache() -> &'static Mutex<HashMap<PathBuf, CachedLocalEmbedder>> {
    static CACHE: OnceLock<Mutex<HashMap<PathBuf, CachedLocalEmbedder>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

#[cfg(feature = "semantic")]
fn load_local_embedder_uncached(home: &Path) -> Result<FastembedEmbedder> {
    let intra_threads = semantic_intra_threads();
    let cache_path = semantic_model_cache_path(home);
    let tokenizer_files = fastembed::TokenizerFiles {
        tokenizer_file: read_model_file(&cache_path, "tokenizer.json")?,
        config_file: read_model_file(&cache_path, "config.json")?,
        special_tokens_map_file: read_model_file(&cache_path, "special_tokens_map.json")?,
        tokenizer_config_file: read_model_file(&cache_path, "tokenizer_config.json")?,
    };
    let mut model = fastembed::UserDefinedEmbeddingModel::new(
        read_model_file(&cache_path, "onnx/model_q4.onnx")?,
        tokenizer_files,
    )
    .with_external_initializer(
        "model_q4.onnx_data".to_string(),
        read_model_file(&cache_path, "onnx/model_q4.onnx_data")?,
    )
    .with_pooling(fastembed::Pooling::Mean)
    .with_quantization(fastembed::QuantizationMode::None);
    model.output_key = Some(fastembed::OutputKey::ByName("sentence_embedding"));

    let text_embedding = fastembed::TextEmbedding::try_new_from_user_defined(
        model,
        fastembed::InitOptionsUserDefined::new()
            .with_max_length(SEMANTIC_EMBED_MAX_LENGTH)
            .with_intra_threads(intra_threads),
    )
    .map_err(|source| {
        Error::SemanticUnavailable(format!("failed to load local embedding model: {source}"))
    })?;

    Ok(FastembedEmbedder {
        model: std::sync::Mutex::new(text_embedding),
        batch_size: SEMANTIC_EMBED_BATCH_SIZE,
        intra_threads,
    })
}

#[cfg(feature = "semantic")]
fn read_model_file(cache_path: &Path, local: &str) -> Result<Vec<u8>> {
    let path = cache_path.join(local);
    fs::read(&path).map_err(|source| Error::Io { path, source })
}

pub fn search_history_page(home: &Path, query: &str, options: SearchOptions) -> Result<SearchPage> {
    if query.trim().is_empty() {
        return Err(Error::Validation("query must not be empty".to_string()));
    }
    let mode_requested = options.mode;
    let semantic_available = semantic_search_available(home);
    let mut mode_applied = match mode_requested {
        SearchMode::Auto if semantic_available => SearchMode::Hybrid,
        SearchMode::Auto => SearchMode::Lexical,
        SearchMode::Lexical => SearchMode::Lexical,
        SearchMode::Hybrid if semantic_available => SearchMode::Hybrid,
        SearchMode::Hybrid => {
            return Err(Error::SemanticUnavailable(
                "local embedding model and vector index are not available; run lexical mode or install the semantic model explicitly".to_string(),
            ))
        }
    };
    if mode_applied == SearchMode::Hybrid {
        match search_history_hybrid_page(home, query, options.clone(), semantic_available) {
            Ok(page) => return Ok(page),
            Err(Error::SemanticUnavailable(_)) if mode_requested == SearchMode::Auto => {
                mode_applied = SearchMode::Lexical;
            }
            Err(error) => return Err(error),
        }
    }
    let query_terms = searchable_terms(query)?;
    let fts_query = quoted_fts_terms(&query_terms);
    let limit = options.limit.clamp(1, MAX_SEARCH_LIMIT);
    let offset = options.offset;
    let max_snippet_chars = options.max_snippet_chars.clamp(1, MAX_SEARCH_SNIPPET_CHARS);
    let raw_fetch_limit = search_overfetch_limit(offset, limit);
    let (mut results, has_more_raw_rows) = lexical_search_ranked_results(
        home,
        &options,
        &query_terms,
        &fts_query,
        raw_fetch_limit,
        max_snippet_chars,
    )?;
    if options.dedupe {
        results = dedupe_ranked_search_results(results)?;
    }

    let total_estimated = if has_more_raw_rows {
        None
    } else {
        Some(results.len())
    };
    let has_more_logical_rows = results.len() > offset.saturating_add(limit);
    let mut page_results = results
        .into_iter()
        .skip(offset)
        .take(limit)
        .map(|ranked| ranked.result)
        .collect::<Vec<_>>();
    if options.include_payload {
        hydrate_search_result_payloads(&mut page_results)?;
    }
    if options.corroborate {
        annotate_search_results_with_corroboration(&mut page_results);
    }
    let returned = page_results.len();
    let continuation = if returned > 0 && (has_more_raw_rows || has_more_logical_rows) {
        Some(SearchContinuation {
            next_offset: offset.saturating_add(returned),
        })
    } else {
        None
    };

    Ok(SearchPage {
        results: page_results,
        truncated: continuation.is_some(),
        returned,
        total_estimated,
        continuation,
        mode_requested,
        mode_applied,
        semantic_available,
        limit_applied: limit,
        offset_applied: offset,
        max_snippet_chars_applied: max_snippet_chars,
        include_payload: options.include_payload,
        include_deltas: options.include_deltas,
        dedupe: options.dedupe,
    })
}

fn lexical_search_ranked_results(
    home: &Path,
    options: &SearchOptions,
    query_terms: &[String],
    fts_query: &str,
    fetch_limit: usize,
    max_snippet_chars: usize,
) -> Result<(Vec<RankedSearchResult>, bool)> {
    let db_path = home.join("index").join("harness.db");
    let conn = open_index(&db_path)?;
    let mut sql = String::from(
        "SELECT
           e.id,
           e.tool,
           e.session_id,
           e.canonical_type,
           e.captured_at,
           -bm25(events_fts, 8.0, 6.0, 4.0, 1.0, 0.5) AS score,
           NULL AS snippet,
           e.searchable_text,
           e.raw_file,
           e.raw_line,
           e.raw_offset,
           e.compaction_state,
           e.cwd,
           e.project_root
         FROM events_fts
         JOIN events e ON e.id = events_fts.rowid
         WHERE events_fts MATCH ?",
    );
    let mut params = vec![SqlValue::Text(fts_query.to_string())];

    if let Some(tool) = options.tool {
        sql.push_str(" AND e.tool = ?");
        params.push(SqlValue::Text(tool.as_str().to_string()));
    }
    if let Some(session_id) = options.session_id.as_deref() {
        sql.push_str(" AND e.session_id = ?");
        params.push(SqlValue::Text(session_id.to_string()));
    }
    if let Some(cwd) = options.cwd.as_deref() {
        sql.push_str(" AND e.cwd = ?");
        params.push(SqlValue::Text(cwd.to_string()));
    }
    if let Some(since) = options.since.as_deref() {
        sql.push_str(" AND e.captured_at >= ?");
        params.push(SqlValue::Text(normalize_date_or_duration(since, "since")?));
    }
    if let Some(canonical_type) = options.canonical_type.as_deref() {
        sql.push_str(" AND e.canonical_type = ?");
        params.push(SqlValue::Text(canonical_type.to_string()));
    }
    if !options.include_deltas {
        sql.push_str(" AND e.canonical_type != 'assistant.delta'");
    }
    if let Some(file) = options.file.as_deref() {
        sql.push_str(
            " AND EXISTS (
                SELECT 1
                FROM event_files ef
                JOIN files f ON f.id = ef.file_id
                WHERE ef.event_id = e.id
                  AND (f.path = ? OR f.path LIKE ?)
              )",
        );
        params.push(SqlValue::Text(file.to_string()));
        params.push(SqlValue::Text(format!("%{file}%")));
    }
    if let Some(command) = options.command.as_deref() {
        sql.push_str(
            " AND EXISTS (
                SELECT 1
                FROM tool_events te
                WHERE te.event_id = e.id
                  AND te.command LIKE ?
              )",
        );
        params.push(SqlValue::Text(format!("%{command}%")));
    }
    sql.push_str(
        " ORDER BY bm25(events_fts, 8.0, 6.0, 4.0, 1.0, 0.5), e.captured_at DESC, e.raw_line ASC
          LIMIT ?",
    );
    params.push(SqlValue::Integer(fetch_limit.saturating_add(1) as i64));

    let mut statement = conn.prepare(&sql).map_err(|source| Error::Sqlite {
        path: db_path.clone(),
        source,
    })?;
    let rows = statement
        .query_map(params_from_iter(params), |row| {
            let tool_text: String = row.get(1)?;
            let searchable_text = row.get::<_, String>(7).unwrap_or_default();
            Ok(RankedSearchResult {
                event_id: row.get(0)?,
                result: SearchResult {
                    tool: Tool::from_str(&tool_text).map_err(|_| rusqlite::Error::InvalidQuery)?,
                    session_id: row.get(2)?,
                    canonical_type: row.get(3)?,
                    timestamp: row.get(4)?,
                    score: row.get(5)?,
                    snippet: match_centered_snippet(
                        row.get::<_, Option<String>>(6)?,
                        searchable_text.clone(),
                        query_terms,
                        max_snippet_chars,
                    ),
                    raw_file: row.get(8)?,
                    raw_line: row.get(9)?,
                    raw_offset: row.get(10)?,
                    compaction_state: row.get(11)?,
                    payload: Value::Null,
                    also_at: Vec::new(),
                    corroboration: None,
                    retrieval_key: sha256_hex(searchable_text.as_bytes()),
                    corroboration_text: searchable_text,
                    cwd: row.get(12)?,
                    project_root: row.get(13)?,
                },
            })
        })
        .map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?;

    let mut results = Vec::new();
    for row in rows {
        results.push(row.map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?);
    }
    let has_more_raw_rows = results.len() > fetch_limit;
    if has_more_raw_rows {
        results.truncate(fetch_limit);
    }
    Ok((results, has_more_raw_rows))
}

fn search_history_hybrid_page(
    home: &Path,
    query: &str,
    options: SearchOptions,
    semantic_available: bool,
) -> Result<SearchPage> {
    let query_terms = searchable_terms(query)?;
    let limit = options.limit.clamp(1, MAX_SEARCH_LIMIT);
    let offset = options.offset;
    let max_snippet_chars = options.max_snippet_chars.clamp(1, MAX_SEARCH_SNIPPET_CHARS);
    let raw_fetch_limit = search_overfetch_limit(offset, limit);
    let fts_query = quoted_fts_terms(&query_terms);

    let (lexical_results, _) = lexical_search_ranked_results(
        home,
        &options,
        &query_terms,
        &fts_query,
        raw_fetch_limit,
        max_snippet_chars,
    )?;
    let vector_results = vector_search_results(
        home,
        query,
        &options,
        raw_fetch_limit,
        &query_terms,
        max_snippet_chars,
    )?;
    let mut results = reciprocal_rank_fuse(lexical_results, vector_results);

    if options.dedupe {
        results = dedupe_ranked_search_results(results)?;
    }
    let total_estimated = Some(results.len());
    let has_more_logical_rows = results.len() > offset.saturating_add(limit);
    let mut page_results = results
        .into_iter()
        .skip(offset)
        .take(limit)
        .map(|ranked| ranked.result)
        .collect::<Vec<_>>();
    if options.include_payload {
        hydrate_search_result_payloads(&mut page_results)?;
    }
    let returned = page_results.len();
    let continuation = if returned > 0 && has_more_logical_rows {
        Some(SearchContinuation {
            next_offset: offset.saturating_add(returned),
        })
    } else {
        None
    };

    Ok(SearchPage {
        results: page_results,
        truncated: continuation.is_some(),
        returned,
        total_estimated,
        continuation,
        mode_requested: options.mode,
        mode_applied: SearchMode::Hybrid,
        semantic_available,
        limit_applied: limit,
        offset_applied: offset,
        max_snippet_chars_applied: max_snippet_chars,
        include_payload: options.include_payload,
        include_deltas: options.include_deltas,
        dedupe: options.dedupe,
    })
}

#[cfg(feature = "semantic")]
fn vector_search_results(
    home: &Path,
    query: &str,
    options: &SearchOptions,
    fetch_limit: usize,
    query_terms: &[String],
    max_snippet_chars: usize,
) -> Result<Vec<RankedSearchResult>> {
    let Some(embedder) = load_local_embedder(home)? else {
        return Err(Error::SemanticUnavailable(
            "local embedding model is not installed".to_string(),
        ));
    };
    let query_vector = embedder.embed_query(query)?;
    let query_blob = vector_to_blob(&query_vector)?;
    let db_path = home.join("index").join("harness.db");
    let conn = open_index(&db_path)?;
    ensure_semantic_vector_schema(&conn, &db_path)?;

    let ctx = VectorQueryContext {
        conn: &conn,
        db_path: &db_path,
        query_blob: &query_blob,
        options,
        query_terms,
        max_snippet_chars,
    };
    let max_vector_k = max_vector_search_k(fetch_limit);
    let mut vector_k = initial_vector_search_k(fetch_limit, options).min(max_vector_k);
    loop {
        let row_limit = vector_search_row_limit(fetch_limit, vector_k);
        let results = vector_search_results_for_k(&ctx, vector_k, row_limit)?;
        let unique = unique_ranked_results_by_event(results);
        if unique.len() >= fetch_limit || vector_k >= max_vector_k {
            return Ok(unique);
        }
        let next_vector_k = vector_k.saturating_mul(2).min(max_vector_k);
        if next_vector_k == vector_k {
            return Ok(unique);
        }
        vector_k = next_vector_k;
    }
}

#[cfg(feature = "semantic")]
/// Loop-invariant inputs to a vector search; only `vector_k`/`row_limit` vary
/// across the adaptive-fetch retries, so the rest travel together as context.
#[cfg(feature = "semantic")]
#[derive(Clone, Copy)]
struct VectorQueryContext<'a> {
    conn: &'a Connection,
    db_path: &'a Path,
    query_blob: &'a [u8],
    options: &'a SearchOptions,
    query_terms: &'a [String],
    max_snippet_chars: usize,
}

#[cfg(feature = "semantic")]
fn vector_search_results_for_k(
    ctx: &VectorQueryContext,
    vector_k: usize,
    row_limit: usize,
) -> Result<Vec<RankedSearchResult>> {
    let VectorQueryContext {
        conn,
        db_path,
        query_blob,
        options,
        query_terms,
        max_snippet_chars,
    } = *ctx;
    let mut sql = String::from(
        "SELECT
           e.id,
           e.tool,
           e.session_id,
           e.canonical_type,
           e.captured_at,
           ve.distance,
           e.searchable_text,
           e.raw_file,
           e.raw_line,
           e.raw_offset,
           e.compaction_state,
           e.cwd,
           e.project_root
         FROM vector_unit_embeddings ve
         JOIN vector_units vu ON vu.id = ve.unit_id
         JOIN events e ON e.id = vu.event_id
         WHERE ve.embedding MATCH ? AND ve.k = ?",
    );
    let mut params = vec![
        SqlValue::Blob(query_blob.to_vec()),
        SqlValue::Integer(vector_k as i64),
    ];

    if let Some(tool) = options.tool {
        sql.push_str(" AND e.tool = ?");
        params.push(SqlValue::Text(tool.as_str().to_string()));
    }
    if let Some(session_id) = options.session_id.as_deref() {
        sql.push_str(" AND e.session_id = ?");
        params.push(SqlValue::Text(session_id.to_string()));
    }
    if let Some(cwd) = options.cwd.as_deref() {
        sql.push_str(" AND e.cwd = ?");
        params.push(SqlValue::Text(cwd.to_string()));
    }
    if let Some(since) = options.since.as_deref() {
        sql.push_str(" AND e.captured_at >= ?");
        params.push(SqlValue::Text(normalize_date_or_duration(since, "since")?));
    }
    if let Some(canonical_type) = options.canonical_type.as_deref() {
        sql.push_str(" AND e.canonical_type = ?");
        params.push(SqlValue::Text(canonical_type.to_string()));
    }
    if !options.include_deltas {
        sql.push_str(" AND e.canonical_type != 'assistant.delta'");
    }
    if let Some(file) = options.file.as_deref() {
        sql.push_str(
            " AND EXISTS (
                SELECT 1
                FROM event_files ef
                JOIN files f ON f.id = ef.file_id
                WHERE ef.event_id = e.id
                  AND (f.path = ? OR f.path LIKE ?)
              )",
        );
        params.push(SqlValue::Text(file.to_string()));
        params.push(SqlValue::Text(format!("%{file}%")));
    }
    if let Some(command) = options.command.as_deref() {
        sql.push_str(
            " AND EXISTS (
                SELECT 1
                FROM tool_events te
                WHERE te.event_id = e.id
                  AND te.command LIKE ?
              )",
        );
        params.push(SqlValue::Text(format!("%{command}%")));
    }
    sql.push_str(" ORDER BY ve.distance LIMIT ?");
    params.push(SqlValue::Integer(row_limit as i64));

    let mut statement = conn.prepare(&sql).map_err(|source| Error::Sqlite {
        path: db_path.to_path_buf(),
        source,
    })?;
    let rows = statement
        .query_map(params_from_iter(params), |row| {
            let tool_text: String = row.get(1)?;
            let searchable_text = row.get::<_, String>(6).unwrap_or_default();
            let distance = row.get::<_, f64>(5)?;
            Ok(RankedSearchResult {
                event_id: row.get(0)?,
                result: SearchResult {
                    tool: Tool::from_str(&tool_text).map_err(|_| rusqlite::Error::InvalidQuery)?,
                    session_id: row.get(2)?,
                    canonical_type: row.get(3)?,
                    timestamp: row.get(4)?,
                    score: 1.0 / (1.0 + distance),
                    snippet: match_centered_snippet(
                        None,
                        searchable_text.clone(),
                        query_terms,
                        max_snippet_chars,
                    ),
                    raw_file: row.get(7)?,
                    raw_line: row.get(8)?,
                    raw_offset: row.get(9)?,
                    compaction_state: row.get(10)?,
                    payload: Value::Null,
                    also_at: Vec::new(),
                    corroboration: None,
                    retrieval_key: sha256_hex(searchable_text.as_bytes()),
                    corroboration_text: searchable_text,
                    cwd: row.get(11)?,
                    project_root: row.get(12)?,
                },
            })
        })
        .map_err(|source| Error::Sqlite {
            path: db_path.to_path_buf(),
            source,
        })?;

    let mut results = Vec::new();
    for row in rows {
        results.push(row.map_err(|source| Error::Sqlite {
            path: db_path.to_path_buf(),
            source,
        })?);
    }
    Ok(results)
}

#[cfg(not(feature = "semantic"))]
fn vector_search_results(
    _home: &Path,
    _query: &str,
    _options: &SearchOptions,
    _fetch_limit: usize,
    _query_terms: &[String],
    _max_snippet_chars: usize,
) -> Result<Vec<RankedSearchResult>> {
    Err(Error::SemanticUnavailable(
        "semantic backend is not available in this build; rebuild with --features semantic"
            .to_string(),
    ))
}

#[cfg(feature = "semantic")]
fn max_vector_search_k(fetch_limit: usize) -> usize {
    fetch_limit
        .clamp(1, MAX_SEARCH_LIMIT * 20)
        .saturating_mul(4)
        .max(1)
}

#[cfg(feature = "semantic")]
fn initial_vector_search_k(fetch_limit: usize, options: &SearchOptions) -> usize {
    let multiplier = if vector_search_filter_count(options) == 0 {
        2
    } else {
        4
    };
    fetch_limit
        .clamp(1, MAX_SEARCH_LIMIT * 20)
        .saturating_mul(multiplier)
        .max(1)
}

#[cfg(feature = "semantic")]
fn vector_search_row_limit(fetch_limit: usize, vector_k: usize) -> usize {
    let vector_k = vector_k.max(1);
    fetch_limit.saturating_mul(2).max(1).min(vector_k)
}

#[cfg(feature = "semantic")]
fn vector_search_filter_count(options: &SearchOptions) -> usize {
    [
        options.tool.is_some(),
        options.session_id.is_some(),
        options.cwd.is_some(),
        options.since.is_some(),
        options.canonical_type.is_some(),
        options.file.is_some(),
        options.command.is_some(),
    ]
    .into_iter()
    .filter(|present| *present)
    .count()
}

fn reciprocal_rank_fuse(
    lexical_results: Vec<RankedSearchResult>,
    vector_results: Vec<RankedSearchResult>,
) -> Vec<RankedSearchResult> {
    const RRF_K: f64 = 60.0;
    let lexical_results = unique_ranked_results_by_event(lexical_results);
    let vector_results = unique_ranked_results_by_event(vector_results);
    let mut fused: HashMap<i64, (RankedSearchResult, f64)> = HashMap::new();

    for (rank, result) in lexical_results.into_iter().enumerate() {
        let key = result.event_id;
        let entry = fused.entry(key).or_insert((result, 0.0));
        entry.1 += 1.0 / (RRF_K + rank as f64 + 1.0);
    }
    for (rank, result) in vector_results.into_iter().enumerate() {
        let key = result.event_id;
        let entry = fused.entry(key).or_insert((result, 0.0));
        entry.1 += 1.0 / (RRF_K + rank as f64 + 1.0);
    }

    let mut results = fused
        .into_values()
        .map(|(mut result, score)| {
            result.result.score = score;
            result
        })
        .collect::<Vec<_>>();
    results.sort_by(|left, right| {
        right
            .result
            .score
            .total_cmp(&left.result.score)
            .then_with(|| right.result.timestamp.cmp(&left.result.timestamp))
            .then_with(|| left.result.raw_line.cmp(&right.result.raw_line))
    });
    results
}

fn unique_ranked_results_by_event(results: Vec<RankedSearchResult>) -> Vec<RankedSearchResult> {
    let mut seen = HashSet::new();
    let mut unique = Vec::new();
    for result in results {
        if seen.insert(result.event_id) {
            unique.push(result);
        }
    }
    unique
}

fn annotate_search_results_with_corroboration(results: &mut [SearchResult]) {
    for result in results {
        result.corroboration = Some(corroborate_text(
            result.cwd.as_deref(),
            result.project_root.as_deref(),
            &result.corroboration_text,
        ));
    }
}

fn corroborate_text(cwd: Option<&str>, project_root: Option<&str>, text: &str) -> Corroboration {
    let candidates = extract_corroboration_candidates(text);
    if candidates.is_empty() {
        return Corroboration {
            repo: None,
            refs: Vec::new(),
        };
    }

    let has_local_refs = candidates
        .iter()
        .any(|candidate| candidate.kind != CorroborationRefKind::Pr);
    let repo_lookup = if has_local_refs {
        locate_git_repo(cwd, project_root)
    } else {
        RepoLookup::NoRepo
    };
    let repo_path = match &repo_lookup {
        RepoLookup::Found(repo) => Some(repo.display().to_string()),
        RepoLookup::NoRepo | RepoLookup::Unknown => None,
    };

    let refs = candidates
        .into_iter()
        .map(|candidate| match candidate.kind {
            CorroborationRefKind::Pr => CorroboratedRef {
                kind: candidate.kind.as_str().to_string(),
                reference: candidate.reference,
                status: "unresolved".to_string(),
                detail: None,
                reason: Some("needs_network".to_string()),
            },
            _ => match &repo_lookup {
                RepoLookup::Found(repo) => resolve_local_ref(repo, cwd, project_root, candidate),
                RepoLookup::NoRepo => CorroboratedRef {
                    kind: candidate.kind.as_str().to_string(),
                    reference: candidate.reference,
                    status: "unresolved".to_string(),
                    detail: None,
                    reason: Some("no_repo".to_string()),
                },
                RepoLookup::Unknown => CorroboratedRef {
                    kind: candidate.kind.as_str().to_string(),
                    reference: candidate.reference,
                    status: "unknown".to_string(),
                    detail: None,
                    reason: Some("git_error".to_string()),
                },
            },
        })
        .collect();

    Corroboration {
        repo: repo_path,
        refs,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum CorroborationRefKind {
    Commit,
    Branch,
    File,
    Pr,
}

impl CorroborationRefKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Commit => "commit",
            Self::Branch => "branch",
            Self::File => "file",
            Self::Pr => "pr",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CorroborationCandidate {
    kind: CorroborationRefKind,
    reference: String,
}

fn extract_corroboration_candidates(text: &str) -> Vec<CorroborationCandidate> {
    let mut candidates = Vec::new();
    let mut seen = HashSet::new();

    let pr_hash = Regex::new(r"(?i)\b(?:PR\s*)?#([0-9]{1,8})\b").expect("valid PR regex");
    for captures in pr_hash.captures_iter(text) {
        push_corroboration_candidate(
            &mut candidates,
            &mut seen,
            CorroborationRefKind::Pr,
            format!("#{}", &captures[1]),
        );
    }
    let pr_pull = Regex::new(r"(?i)\bpull/([0-9]{1,8})\b").expect("valid pull regex");
    for captures in pr_pull.captures_iter(text) {
        push_corroboration_candidate(
            &mut candidates,
            &mut seen,
            CorroborationRefKind::Pr,
            format!("#{}", &captures[1]),
        );
    }

    let commit =
        Regex::new(r"(?i)(^|[^0-9a-f])([0-9a-f]{7,40})([^0-9a-f]|$)").expect("valid commit regex");
    for captures in commit.captures_iter(text) {
        push_corroboration_candidate(
            &mut candidates,
            &mut seen,
            CorroborationRefKind::Commit,
            captures[2].to_string(),
        );
    }

    for pattern in [
        r"(?i)\bbranch\s+([A-Za-z0-9][A-Za-z0-9._/\-]{0,200})",
        r"(?i)\bgit\s+(?:checkout|switch)\s+(?:--track\s+)?(?:-c\s+)?([A-Za-z0-9][A-Za-z0-9._/\-]{0,200})",
    ] {
        let regex = Regex::new(pattern).expect("valid branch regex");
        for captures in regex.captures_iter(text) {
            if let Some(reference) = clean_reference_token(&captures[1]) {
                push_corroboration_candidate(
                    &mut candidates,
                    &mut seen,
                    CorroborationRefKind::Branch,
                    reference,
                );
            }
        }
    }
    let origin_branch =
        Regex::new(r"\borigin/([A-Za-z0-9][A-Za-z0-9._/\-]{0,200})").expect("valid origin regex");
    for captures in origin_branch.captures_iter(text) {
        if let Some(reference) = clean_reference_token(&format!("origin/{}", &captures[1])) {
            push_corroboration_candidate(
                &mut candidates,
                &mut seen,
                CorroborationRefKind::Branch,
                reference,
            );
        }
    }

    for pattern in [
        r#"(?m)(?:^|[\s("'`])(/(?:[A-Za-z0-9._-]+/)+[A-Za-z0-9._-]+\.[A-Za-z][A-Za-z0-9._-]{0,20})"#,
        r"\b((?:[A-Za-z0-9._-]+/)+[A-Za-z0-9._-]+\.[A-Za-z][A-Za-z0-9._-]{0,20})\b",
    ] {
        let regex = Regex::new(pattern).expect("valid file regex");
        for captures in regex.captures_iter(text) {
            if let Some(reference) = clean_file_reference(&captures[1]) {
                push_corroboration_candidate(
                    &mut candidates,
                    &mut seen,
                    CorroborationRefKind::File,
                    reference,
                );
            }
        }
    }

    candidates
}

fn push_corroboration_candidate(
    candidates: &mut Vec<CorroborationCandidate>,
    seen: &mut HashSet<(CorroborationRefKind, String)>,
    kind: CorroborationRefKind,
    reference: String,
) {
    if reference.trim().is_empty() {
        return;
    }
    let key = (kind, reference.clone());
    if seen.insert(key) {
        candidates.push(CorroborationCandidate { kind, reference });
    }
}

fn clean_reference_token(value: &str) -> Option<String> {
    let trimmed = value.trim_matches(reference_boundary_character).trim();
    if trimmed.is_empty()
        || trimmed.starts_with('-')
        || trimmed.ends_with('/')
        || trimmed.contains("..")
        || trimmed.contains("://")
    {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn clean_file_reference(value: &str) -> Option<String> {
    let reference = clean_reference_token(value)?;
    if reference.starts_with("origin/")
        || reference.starts_with("http")
        || reference.contains('#')
        || reference.len() > 512
    {
        return None;
    }
    Some(reference)
}

fn reference_boundary_character(character: char) -> bool {
    matches!(
        character,
        '"' | '\'' | '`' | ')' | ']' | '}' | '>' | ',' | ';' | ':' | '!' | '.'
    )
}

enum RepoLookup {
    Found(PathBuf),
    NoRepo,
    Unknown,
}

fn locate_git_repo(cwd: Option<&str>, project_root: Option<&str>) -> RepoLookup {
    let Some(start) = repo_start_path(cwd, project_root) else {
        return RepoLookup::NoRepo;
    };
    if !start.exists() {
        return RepoLookup::NoRepo;
    }
    match run_git_read(&start, &["rev-parse", "--show-toplevel"]) {
        GitOutcome::Success(stdout) => {
            let repo = stdout.lines().next().unwrap_or("").trim();
            if repo.is_empty() {
                RepoLookup::Unknown
            } else {
                let repo = PathBuf::from(repo);
                RepoLookup::Found(fs::canonicalize(&repo).unwrap_or(repo))
            }
        }
        GitOutcome::NonZero => RepoLookup::NoRepo,
        GitOutcome::Failed => RepoLookup::Unknown,
    }
}

fn repo_start_path(cwd: Option<&str>, project_root: Option<&str>) -> Option<PathBuf> {
    cwd.filter(|value| !value.trim().is_empty())
        .or_else(|| project_root.filter(|value| !value.trim().is_empty()))
        .map(PathBuf::from)
}

fn resolve_local_ref(
    repo: &Path,
    cwd: Option<&str>,
    project_root: Option<&str>,
    candidate: CorroborationCandidate,
) -> CorroboratedRef {
    match candidate.kind {
        CorroborationRefKind::Commit => resolve_commit_ref(repo, candidate),
        CorroborationRefKind::Branch => resolve_branch_ref(repo, candidate),
        CorroborationRefKind::File => resolve_file_ref(repo, cwd, project_root, candidate),
        CorroborationRefKind::Pr => unreachable!("PR refs do not resolve locally"),
    }
}

fn resolve_commit_ref(repo: &Path, candidate: CorroborationCandidate) -> CorroboratedRef {
    let commitish = format!("{}^{{commit}}", candidate.reference);
    match run_git_read(repo, &["cat-file", "-e", &commitish]) {
        GitOutcome::Success(_) => {
            let detail = match run_git_read(
                repo,
                &[
                    "log",
                    "-1",
                    "--format=%h %s",
                    "--no-show-signature",
                    &candidate.reference,
                ],
            ) {
                GitOutcome::Success(stdout) => stdout
                    .lines()
                    .next()
                    .map(str::trim)
                    .filter(|line| !line.is_empty())
                    .map(ToOwned::to_owned),
                GitOutcome::NonZero | GitOutcome::Failed => None,
            };
            CorroboratedRef {
                kind: candidate.kind.as_str().to_string(),
                reference: candidate.reference,
                status: "present".to_string(),
                detail,
                reason: None,
            }
        }
        GitOutcome::NonZero => CorroboratedRef {
            kind: candidate.kind.as_str().to_string(),
            reference: candidate.reference,
            status: "missing".to_string(),
            detail: None,
            reason: None,
        },
        GitOutcome::Failed => CorroboratedRef {
            kind: candidate.kind.as_str().to_string(),
            reference: candidate.reference,
            status: "unknown".to_string(),
            detail: None,
            reason: Some("git_error".to_string()),
        },
    }
}

fn resolve_branch_ref(repo: &Path, candidate: CorroborationCandidate) -> CorroboratedRef {
    let full_ref = if let Some(remote_branch) = candidate.reference.strip_prefix("origin/") {
        format!("refs/remotes/origin/{remote_branch}")
    } else {
        format!("refs/heads/{}", candidate.reference)
    };
    match run_git_read(repo, &["rev-parse", "--verify", "--quiet", &full_ref]) {
        GitOutcome::Success(_) => CorroboratedRef {
            kind: candidate.kind.as_str().to_string(),
            reference: candidate.reference,
            status: "present".to_string(),
            detail: None,
            reason: None,
        },
        GitOutcome::NonZero => CorroboratedRef {
            kind: candidate.kind.as_str().to_string(),
            reference: candidate.reference,
            status: "missing".to_string(),
            detail: None,
            reason: None,
        },
        GitOutcome::Failed => CorroboratedRef {
            kind: candidate.kind.as_str().to_string(),
            reference: candidate.reference,
            status: "unknown".to_string(),
            detail: None,
            reason: Some("git_error".to_string()),
        },
    }
}

fn resolve_file_ref(
    repo: &Path,
    cwd: Option<&str>,
    project_root: Option<&str>,
    candidate: CorroborationCandidate,
) -> CorroboratedRef {
    let Some(relative_path) =
        candidate_file_repo_path(repo, cwd, project_root, &candidate.reference)
    else {
        return CorroboratedRef {
            kind: candidate.kind.as_str().to_string(),
            reference: candidate.reference,
            status: "missing".to_string(),
            detail: None,
            reason: None,
        };
    };
    let relative_text = relative_path.to_string_lossy().to_string();
    let on_disk = repo.join(&relative_path).exists();
    match run_git_read(repo, &["ls-files", "--error-unmatch", "--", &relative_text]) {
        GitOutcome::Success(_) => CorroboratedRef {
            kind: candidate.kind.as_str().to_string(),
            reference: candidate.reference,
            status: "present".to_string(),
            detail: Some("tracked".to_string()),
            reason: None,
        },
        GitOutcome::NonZero if on_disk => CorroboratedRef {
            kind: candidate.kind.as_str().to_string(),
            reference: candidate.reference,
            status: "untracked".to_string(),
            detail: None,
            reason: None,
        },
        GitOutcome::NonZero => CorroboratedRef {
            kind: candidate.kind.as_str().to_string(),
            reference: candidate.reference,
            status: "missing".to_string(),
            detail: None,
            reason: None,
        },
        GitOutcome::Failed => CorroboratedRef {
            kind: candidate.kind.as_str().to_string(),
            reference: candidate.reference,
            status: "unknown".to_string(),
            detail: None,
            reason: Some("git_error".to_string()),
        },
    }
}

fn candidate_file_repo_path(
    repo: &Path,
    cwd: Option<&str>,
    project_root: Option<&str>,
    reference: &str,
) -> Option<PathBuf> {
    let reference_path = PathBuf::from(reference);
    if reference_path.is_absolute() {
        return path_under_repo(repo, &reference_path);
    }

    for base in [
        cwd.and_then(|value| (!value.trim().is_empty()).then_some(PathBuf::from(value))),
        project_root.and_then(|value| (!value.trim().is_empty()).then_some(PathBuf::from(value))),
        Some(repo.to_path_buf()),
    ]
    .into_iter()
    .flatten()
    {
        if let Some(path) = path_under_repo(repo, &base.join(reference)) {
            return Some(path);
        }
    }
    None
}

fn path_under_repo(repo: &Path, path: &Path) -> Option<PathBuf> {
    let normalized = normalize_path(path);
    let normalized_repo = normalize_path(repo);
    normalized
        .strip_prefix(&normalized_repo)
        .ok()
        .filter(|relative| !relative.as_os_str().is_empty())
        .map(Path::to_path_buf)
}

fn normalize_path(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                normalized.pop();
            }
            _ => normalized.push(component.as_os_str()),
        }
    }
    normalized
}

enum GitOutcome {
    Success(String),
    NonZero,
    Failed,
}

fn run_git_read(repo: &Path, args: &[&str]) -> GitOutcome {
    record_git_invocation(args);

    let mut command = ProcessCommand::new(git_binary());
    command
        .arg("-c")
        .arg("core.hooksPath=/dev/null")
        .arg("-c")
        .arg("pager.branch=false")
        .arg("-C")
        .arg(repo)
        .arg("--no-pager")
        .args(args)
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GIT_PAGER", "cat")
        .env("PAGER", "cat")
        .env("GIT_OPTIONAL_LOCKS", "0")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let Ok(mut child) = command.spawn() else {
        return GitOutcome::Failed;
    };
    let started = Instant::now();
    let timeout = StdDuration::from_secs(2);
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if started.elapsed() < timeout => {
                std::thread::sleep(StdDuration::from_millis(10));
            }
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait_with_output();
                return GitOutcome::Failed;
            }
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait_with_output();
                return GitOutcome::Failed;
            }
        }
    }

    match child.wait_with_output() {
        Ok(output) if output.status.success() => {
            GitOutcome::Success(String::from_utf8_lossy(&output.stdout).to_string())
        }
        Ok(_) => GitOutcome::NonZero,
        Err(_) => GitOutcome::Failed,
    }
}

fn git_binary() -> String {
    std::env::var("NABU_GIT").unwrap_or_else(|_| "git".to_string())
}

#[cfg(test)]
fn record_git_invocation(args: &[&str]) {
    git_invocations()
        .lock()
        .unwrap()
        .push(args.iter().map(|arg| (*arg).to_string()).collect());
}

#[cfg(not(test))]
fn record_git_invocation(_args: &[&str]) {}

#[cfg(test)]
fn git_invocations() -> &'static std::sync::Mutex<Vec<Vec<String>>> {
    static INVOCATIONS: std::sync::OnceLock<std::sync::Mutex<Vec<Vec<String>>>> =
        std::sync::OnceLock::new();
    INVOCATIONS.get_or_init(|| std::sync::Mutex::new(Vec::new()))
}

fn search_overfetch_limit(offset: usize, limit: usize) -> usize {
    let requested_window = offset.saturating_add(limit);
    let extra = requested_window.min(500).max(limit);
    requested_window.saturating_add(extra)
}

fn bounded_snippet(snippet: String, max_chars: usize) -> String {
    truncate_chars(snippet.trim().to_string(), max_chars)
}

fn match_centered_snippet(
    sqlite_snippet: Option<String>,
    searchable_text: String,
    query_terms: &[String],
    max_chars: usize,
) -> String {
    if let Some(snippet) = sqlite_snippet.filter(|snippet| !snippet.trim().is_empty()) {
        return bounded_snippet(snippet, max_chars);
    }
    if searchable_text.chars().count() <= max_chars {
        return searchable_text.trim().to_string();
    }
    let lower_text = searchable_text.to_lowercase();
    let first_match = query_terms
        .iter()
        .filter_map(|term| lower_text.find(&term.to_lowercase()))
        .min()
        .unwrap_or(0);
    let half_window = max_chars.saturating_div(2);
    let mut start = first_match.saturating_sub(half_window);
    while start > 0 && !searchable_text.is_char_boundary(start) {
        start -= 1;
    }
    let mut end = start.saturating_add(max_chars).min(searchable_text.len());
    while end > start && !searchable_text.is_char_boundary(end) {
        end -= 1;
    }
    searchable_text[start..end].trim().to_string()
}

fn truncate_chars(mut value: String, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value;
    }
    let mut cutoff = 0usize;
    for (count, (index, character)) in value.char_indices().enumerate() {
        if count == max_chars {
            break;
        }
        cutoff = index + character.len_utf8();
    }
    value.truncate(cutoff);
    value
}

fn dedupe_ranked_search_results(
    results: Vec<RankedSearchResult>,
) -> Result<Vec<RankedSearchResult>> {
    let mut seen: HashMap<(String, String, String), usize> = HashMap::new();
    let mut deduped: Vec<RankedSearchResult> = Vec::new();
    for result in results {
        let key = retrieval_twin_key(&result.result);
        if let Some(existing) = seen.get(&key).copied() {
            deduped[existing]
                .result
                .also_at
                .push(result.result.raw_line);
        } else {
            seen.insert(key, deduped.len());
            deduped.push(result);
        }
    }
    Ok(deduped)
}

fn retrieval_twin_key(result: &SearchResult) -> (String, String, String) {
    (
        result.session_id.clone(),
        result.canonical_type.clone(),
        result.retrieval_key.clone(),
    )
}

fn searchable_terms(query: &str) -> Result<Vec<String>> {
    let mut terms = Vec::new();
    let mut current = String::new();

    for character in query.chars() {
        if character.is_alphanumeric() || character == '_' {
            current.push(character);
        } else if !current.is_empty() {
            terms.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        terms.push(current);
    }

    if terms.is_empty() {
        return Err(Error::Validation(
            "query must contain searchable text".to_string(),
        ));
    }

    Ok(terms)
}

fn quoted_fts_terms(terms: &[String]) -> String {
    terms
        .iter()
        .map(|term| format!("\"{}\"", term.replace('"', "\"\"")))
        .collect::<Vec<_>>()
        .join(" AND ")
}

pub fn session_events(home: &Path, tool: Tool, session_id: &str) -> Result<Vec<StoredEvent>> {
    let db_path = home.join("index").join("harness.db");
    let conn = open_index(&db_path)?;
    let mut statement = conn
        .prepare(
            "SELECT
               tool,
               session_id,
               canonical_type,
               captured_at,
               searchable_text,
               raw_file,
               raw_line,
               raw_offset,
               cwd,
               project_root
             FROM events
             WHERE tool = ?1 AND session_id = ?2
             ORDER BY raw_line, raw_offset",
        )
        .map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?;
    let rows = statement
        .query_map((tool.as_str(), session_id), |row| {
            let tool_text: String = row.get(0)?;
            Ok(StoredEvent {
                tool: Tool::from_str(&tool_text).map_err(|_| rusqlite::Error::InvalidQuery)?,
                session_id: row.get(1)?,
                canonical_type: row.get(2)?,
                timestamp: row.get(3)?,
                text: row.get(4)?,
                raw_file: row.get(5)?,
                raw_line: row.get(6)?,
                raw_offset: row.get(7)?,
                corroboration: None,
                cwd: row.get(8)?,
                project_root: row.get(9)?,
            })
        })
        .map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?;

    let mut events = Vec::new();
    for row in rows {
        events.push(row.map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?);
    }
    Ok(events)
}

pub fn list_sessions(
    home: &Path,
    tool: Option<Tool>,
    cwd: Option<&str>,
    since: Option<&str>,
    limit: usize,
) -> Result<Vec<SessionSummary>> {
    let db_path = home.join("index").join("harness.db");
    let conn = open_index(&db_path)?;
    let mut sql = String::from(
        "SELECT
           tool,
           session_id,
           project_root,
           cwd,
           started_at,
           updated_at,
           event_count,
           message_count,
           compaction_count,
           raw_file
         FROM sessions
         WHERE 1 = 1",
    );
    let mut params = Vec::new();

    if let Some(tool) = tool {
        sql.push_str(" AND tool = ?");
        params.push(SqlValue::Text(tool.as_str().to_string()));
    }
    if let Some(cwd) = cwd {
        sql.push_str(" AND cwd = ?");
        params.push(SqlValue::Text(cwd.to_string()));
    }
    if let Some(since) = since {
        sql.push_str(" AND updated_at >= ?");
        params.push(SqlValue::Text(normalize_date_or_duration(since, "since")?));
    }
    sql.push_str(" ORDER BY updated_at DESC LIMIT ?");
    params.push(SqlValue::Integer(limit.clamp(1, 100) as i64));

    let mut statement = conn.prepare(&sql).map_err(|source| Error::Sqlite {
        path: db_path.clone(),
        source,
    })?;
    let rows = statement
        .query_map(params_from_iter(params), |row| {
            let tool_text: String = row.get(0)?;
            Ok(SessionSummary {
                tool: Tool::from_str(&tool_text).map_err(|_| rusqlite::Error::InvalidQuery)?,
                session_id: row.get(1)?,
                project_root: row.get(2)?,
                cwd: row.get(3)?,
                started_at: row.get(4)?,
                updated_at: row.get(5)?,
                event_count: row.get(6)?,
                message_count: row.get(7)?,
                compaction_count: row.get(8)?,
                raw_file: row.get(9)?,
            })
        })
        .map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?;

    let mut sessions = Vec::new();
    for row in rows {
        sessions.push(row.map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?);
    }
    Ok(sessions)
}

pub fn get_session_page(
    home: &Path,
    tool: Tool,
    session_id: &str,
    options: SessionOptions,
) -> Result<SessionPage> {
    let raw_file = session_raw_file(home, tool, session_id)?;
    let db_path = home.join("index").join("harness.db");
    let conn = open_index(&db_path)?;
    let limit = options.limit_events.clamp(1, MAX_SESSION_LIMIT);
    let before = options.before.clamp(0, MAX_CONTEXT_EVENTS_PER_SIDE);
    let after = options.after.clamp(0, MAX_CONTEXT_EVENTS_PER_SIDE);
    let mut sql = String::from(
        "SELECT
               tool,
               session_id,
               canonical_type,
               captured_at,
               searchable_text,
               raw_file,
               raw_line,
               raw_offset,
               cwd,
               project_root
             FROM events
             WHERE tool = ? AND session_id = ?",
    );
    let mut params = vec![
        SqlValue::Text(tool.as_str().to_string()),
        SqlValue::Text(session_id.to_string()),
    ];
    let mode;
    let mut limit_for_query = None;
    if let Some(around_raw_line) = options.around_raw_line {
        mode = "window".to_string();
        let start_line = around_raw_line.saturating_sub(before as i64).max(1);
        let end_line = around_raw_line.saturating_add(after as i64);
        sql.push_str(" AND raw_line >= ? AND raw_line <= ?");
        params.push(SqlValue::Integer(start_line));
        params.push(SqlValue::Integer(end_line));
    } else {
        mode = "page".to_string();
        let after_raw_line = options.after_raw_line.unwrap_or(0);
        sql.push_str(" AND raw_line > ?");
        params.push(SqlValue::Integer(after_raw_line));
        limit_for_query = Some(limit.saturating_add(1));
    }
    if !options.include_deltas {
        sql.push_str(" AND canonical_type != 'assistant.delta'");
    }
    if let Some(canonical_type) = options.canonical_type.as_deref() {
        sql.push_str(" AND canonical_type = ?");
        params.push(SqlValue::Text(canonical_type.to_string()));
    }
    sql.push_str(" ORDER BY raw_line, raw_offset");
    if let Some(limit_for_query) = limit_for_query {
        sql.push_str(" LIMIT ?");
        params.push(SqlValue::Integer(limit_for_query as i64));
    }

    let mut statement = conn.prepare(&sql).map_err(|source| Error::Sqlite {
        path: db_path.clone(),
        source,
    })?;
    let rows = statement
        .query_map(params_from_iter(params), |row| {
            let tool_text: String = row.get(0)?;
            Ok(StoredEvent {
                tool: Tool::from_str(&tool_text).map_err(|_| rusqlite::Error::InvalidQuery)?,
                session_id: row.get(1)?,
                canonical_type: row.get(2)?,
                timestamp: row.get(3)?,
                text: row.get(4)?,
                raw_file: row.get(5)?,
                raw_line: row.get(6)?,
                raw_offset: row.get(7)?,
                corroboration: None,
                cwd: row.get(8)?,
                project_root: row.get(9)?,
            })
        })
        .map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?;

    let mut events = Vec::new();
    for row in rows {
        let mut event = row.map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?;
        if options.redact {
            event.text = redact_text(&event.text);
        }
        if options.corroborate {
            event.corroboration = Some(corroborate_text(
                event.cwd.as_deref(),
                event.project_root.as_deref(),
                &event.text,
            ));
        }
        events.push(event);
    }
    let truncated = options.around_raw_line.is_none() && events.len() > limit;
    if truncated {
        events.truncate(limit);
    }
    let next_after_raw_line = if truncated {
        events.last().map(|event| event.raw_line)
    } else {
        None
    };

    Ok(SessionPage {
        tool,
        session_id: session_id.to_string(),
        raw_file,
        events,
        truncated,
        next_after_raw_line,
        mode,
        limit_events_applied: if options.around_raw_line.is_none() {
            Some(limit)
        } else {
            None
        },
        after_raw_line: if options.around_raw_line.is_none() {
            Some(options.after_raw_line.unwrap_or(0))
        } else {
            None
        },
        around_raw_line: options.around_raw_line,
        before_applied: options.around_raw_line.map(|_| before),
        after_applied: options.around_raw_line.map(|_| after),
        include_deltas: options.include_deltas,
        canonical_type: options.canonical_type,
    })
}

pub fn get_event_by_pointer(
    home: &Path,
    tool: Tool,
    session_id: &str,
    raw_line: Option<i64>,
    raw_offset: Option<i64>,
    redact: bool,
) -> Result<EventPointer> {
    get_event_by_pointer_with_options(
        home,
        tool,
        session_id,
        raw_line,
        raw_offset,
        EventOptions {
            redact,
            corroborate: false,
        },
    )
}

pub fn get_event_by_pointer_with_options(
    home: &Path,
    tool: Tool,
    session_id: &str,
    raw_line: Option<i64>,
    raw_offset: Option<i64>,
    options: EventOptions,
) -> Result<EventPointer> {
    if raw_line.is_none() && raw_offset.is_none() {
        return Err(Error::Validation(
            "raw_line or raw_offset is required".to_string(),
        ));
    }
    let db_path = home.join("index").join("harness.db");
    let conn = open_index(&db_path)?;
    let mut sql = String::from(
        "SELECT raw_file, raw_line, raw_offset, searchable_text, cwd, project_root
         FROM events
         WHERE tool = ? AND session_id = ?",
    );
    let mut params = vec![
        SqlValue::Text(tool.as_str().to_string()),
        SqlValue::Text(session_id.to_string()),
    ];
    if let Some(raw_line) = raw_line {
        sql.push_str(" AND raw_line = ?");
        params.push(SqlValue::Integer(raw_line));
    }
    if let Some(raw_offset) = raw_offset {
        sql.push_str(" AND raw_offset = ?");
        params.push(SqlValue::Integer(raw_offset));
    }
    sql.push_str(" ORDER BY raw_line, raw_offset LIMIT 1");

    let pointer = conn
        .query_row(&sql, params_from_iter(params), |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, Option<i64>>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, Option<String>>(4)?,
                row.get::<_, Option<String>>(5)?,
            ))
        })
        .optional()
        .map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?
        .ok_or_else(|| {
            Error::Validation(format!(
                "event not found for {}:{}",
                tool.as_str(),
                session_id
            ))
        })?;

    let (raw_file, raw_line, raw_offset, searchable_text, cwd, project_root) = pointer;
    let mut envelope = raw_envelope_for_pointer(&raw_file, raw_line, raw_offset)?;
    let mut searchable_text = searchable_text;
    if options.redact {
        envelope.payload = redact_json_value(envelope.payload);
        searchable_text = redact_text(&searchable_text);
    }
    let corroboration = options
        .corroborate
        .then(|| corroborate_text(cwd.as_deref(), project_root.as_deref(), &searchable_text));

    Ok(EventPointer {
        envelope,
        searchable_text,
        raw_file,
        raw_line,
        raw_offset,
        corroboration,
    })
}

pub fn latest_event(home: &Path, tool: Tool) -> Result<Option<StoredEvent>> {
    let db_path = home.join("index").join("harness.db");
    let conn = open_index(&db_path)?;
    let mut statement = conn
        .prepare(
            "SELECT
               tool,
               session_id,
               canonical_type,
               captured_at,
               searchable_text,
               raw_file,
               raw_line,
               raw_offset,
               cwd,
               project_root
             FROM events
             WHERE tool = ?1
             ORDER BY captured_at DESC, id DESC
             LIMIT 1",
        )
        .map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?;
    let mut rows = statement
        .query([tool.as_str()])
        .map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?;
    let Some(row) = rows.next().map_err(|source| Error::Sqlite {
        path: db_path.clone(),
        source,
    })?
    else {
        return Ok(None);
    };
    let tool_text: String = row.get(0).map_err(|source| Error::Sqlite {
        path: db_path.clone(),
        source,
    })?;
    Ok(Some(StoredEvent {
        tool: Tool::from_str(&tool_text)
            .map_err(|_| Error::Validation("invalid tool".to_string()))?,
        session_id: row.get(1).map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?,
        canonical_type: row.get(2).map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?,
        timestamp: row.get(3).map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?,
        text: row.get(4).map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?,
        raw_file: row.get(5).map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?,
        raw_line: row.get(6).map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?,
        raw_offset: row.get(7).map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?,
        corroboration: None,
        cwd: row.get(8).map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?,
        project_root: row.get(9).map_err(|source| Error::Sqlite {
            path: db_path,
            source,
        })?,
    }))
}

pub fn export_session_jsonl(home: &Path, tool: Tool, session_id: &str) -> Result<String> {
    export_session_jsonl_with_options(home, tool, session_id, false)
}

pub fn export_session_jsonl_with_options(
    home: &Path,
    tool: Tool,
    session_id: &str,
    redact: bool,
) -> Result<String> {
    let path = canonical_raw_path(home, tool, session_id);
    let mut content = String::new();
    File::open(&path)
        .map_err(|source| Error::Io {
            path: path.clone(),
            source,
        })?
        .read_to_string(&mut content)
        .map_err(|source| Error::Io { path, source })?;
    if redact {
        Ok(redact_text(&content))
    } else {
        Ok(content)
    }
}

pub fn export_session_markdown(home: &Path, tool: Tool, session_id: &str) -> Result<String> {
    export_session_markdown_with_options(home, tool, session_id, false)
}

pub fn export_session_markdown_with_options(
    home: &Path,
    tool: Tool,
    session_id: &str,
    redact: bool,
) -> Result<String> {
    let mut output = if redact {
        String::from("# nabu Session Export\n\nSensitivity: redacted export.\n\n")
    } else {
        String::from("# nabu Session Export\n\nSensitivity: this export is not redacted.\n\n")
    };
    for event in session_events(home, tool, session_id)? {
        let text = if redact {
            redact_text(&event.text)
        } else {
            event.text
        };
        output.push_str(&format!(
            "## {} {}:{}\n\n{}\n\n",
            event.canonical_type, event.raw_file, event.raw_line, text
        ));
    }
    Ok(output)
}

pub fn redact_export_text(text: &str) -> String {
    redact_text(text)
}

pub fn redact_export_json(value: Value) -> Value {
    redact_json_value(value)
}

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

fn hydrate_search_result_payloads(results: &mut [SearchResult]) -> Result<()> {
    let mut grouped = BTreeMap::<String, Vec<usize>>::new();
    for (index, result) in results.iter().enumerate() {
        grouped
            .entry(result.raw_file.clone())
            .or_default()
            .push(index);
    }

    for (raw_file, mut indexes) in grouped {
        indexes.sort_by_key(|index| {
            (
                results[*index].raw_offset.unwrap_or(i64::MAX),
                results[*index].raw_line,
            )
        });
        let raw_path = PathBuf::from(&raw_file);
        let mut offset_reader = None;
        for index in indexes {
            let raw_line = results[index].raw_line;
            let raw_offset = results[index].raw_offset;
            let envelope = if let Some(raw_offset) = raw_offset {
                if offset_reader.is_none() {
                    offset_reader = Some(open_raw_offset_reader(&raw_path)?);
                }
                match read_raw_envelope_at_offset(
                    &raw_path,
                    offset_reader.as_mut().expect("offset reader initialized"),
                    raw_offset,
                )? {
                    Some(envelope) => envelope,
                    None => raw_envelope_for_line_scan(&raw_path, raw_line)?,
                }
            } else {
                raw_envelope_for_line_scan(&raw_path, raw_line)?
            };
            results[index].payload = resolved_payload_for_envelope(&raw_path, &envelope)?;
        }
    }
    Ok(())
}

fn rewrite_raw_file_after(path: &Path, before: &str) -> Result<bool> {
    let content = fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut kept = Vec::new();
    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let envelope: EventEnvelope = serde_json::from_str(line)?;
        if envelope.captured_at.as_str() >= before {
            kept.push(line.to_string());
        }
    }

    if kept.is_empty() {
        fs::remove_file(path).map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
        remove_dedupe_sidecar_for_raw_file(path)?;
        return Ok(true);
    }

    let mut rewritten = kept.join("\n");
    rewritten.push('\n');
    fs::write(path, rewritten).map_err(|source| Error::Io {
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
    recalculate_session_counts(&tx)?;
    tx.commit().map_err(|source| Error::Sqlite {
        path: db_path.clone(),
        source,
    })?;

    Ok(fts_rows.len())
}

fn redact_json_value(value: Value) -> Value {
    match value {
        Value::String(text) => Value::String(redact_text(&text)),
        Value::Array(values) => Value::Array(values.into_iter().map(redact_json_value).collect()),
        Value::Object(map) => Value::Object(
            map.into_iter()
                .map(|(key, value)| {
                    if is_sensitive_key(&key) {
                        (key, Value::String("[REDACTED:ENV_VALUE]".to_string()))
                    } else {
                        (key, redact_json_value(value))
                    }
                })
                .collect(),
        ),
        other => other,
    }
}

fn redact_text(text: &str) -> String {
    let rules = [
        (
            r"(?s)-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
            "[REDACTED:PRIVATE_KEY]",
        ),
        (
            r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{16,}",
            "Bearer [REDACTED:BEARER_TOKEN]",
        ),
        (r"\bsk-[A-Za-z0-9_-]{20,}\b", "[REDACTED:API_KEY]"),
        (r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b", "[REDACTED:API_KEY]"),
        (r"\bgithub_pat_[A-Za-z0-9_]{20,}\b", "[REDACTED:API_KEY]"),
        (r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b", "[REDACTED:API_KEY]"),
        (r"\bAKIA[0-9A-Z]{16}\b", "[REDACTED:API_KEY]"),
    ];
    let mut redacted = text.to_string();
    for (pattern, replacement) in rules {
        redacted = Regex::new(pattern)
            .expect("valid redaction regex")
            .replace_all(&redacted, replacement)
            .into_owned();
    }
    Regex::new(
        r##"(?im)^([A-Z0-9_]*(API|TOKEN|SECRET|KEY|PASSWORD)[A-Z0-9_]*\s*=\s*)(['"]?)[^\s'"#]{8,}(['"]?)"##,
    )
    .expect("valid env redaction regex")
    .replace_all(&redacted, |captures: &Captures<'_>| {
        format!("{}[REDACTED:ENV_VALUE]", &captures[1])
    })
    .into_owned()
}

fn is_sensitive_key(key: &str) -> bool {
    let normalized = key.to_ascii_lowercase();
    normalized.contains("api_key")
        || normalized.contains("apikey")
        || normalized.contains("access_token")
        || normalized.contains("auth_token")
        || normalized.contains("bearer")
        || normalized.contains("secret")
        || normalized.contains("password")
        || normalized.contains("private_key")
        || normalized.ends_with("_key")
        || normalized.ends_with("token")
}

pub fn backfill(
    home: &Path,
    selection: Option<Tool>,
    source_root: &Path,
) -> Result<BackfillReport> {
    backfill_since(home, selection, source_root, None)
}

pub fn backfill_since(
    home: &Path,
    selection: Option<Tool>,
    source_root: &Path,
    since: Option<&str>,
) -> Result<BackfillReport> {
    backfill_since_with_progress(home, selection, source_root, since, |_| {})
}

/// A source file discovered during the native-store scan can be deleted or
/// rotated by the live tool before backfill reads it (`os error 2`). Such a
/// vanished file must be skipped, never treated as fatal — one missing session
/// must not abort a backfill over the whole store.
fn is_vanished_source(error: &Error) -> bool {
    matches!(error, Error::Io { source, .. } if source.kind() == std::io::ErrorKind::NotFound)
}

pub fn backfill_since_with_progress<F>(
    home: &Path,
    selection: Option<Tool>,
    source_root: &Path,
    since: Option<&str>,
    progress: F,
) -> Result<BackfillReport>
where
    F: Fn(BackfillProgress) + Sync,
{
    init_home(home)?;
    let since_threshold = since.map(parse_since_threshold).transpose()?;
    let mut report = empty_backfill_report();

    let tools: Vec<Tool> = match selection {
        Some(tool) => vec![tool],
        None => Tool::all().to_vec(),
    };

    for tool in tools {
        let tool_root = backfill_tool_root(source_root, tool);
        if !tool_root.exists() {
            continue;
        }
        let mut files = Vec::new();
        collect_backfill_files(tool, &tool_root, &mut files)?;
        files.sort();
        let parse_context = backfill_parse_context(tool, &tool_root, &files)?;
        let files = filter_backfill_files_by_since(files, since_threshold)?;
        let total_files = files.len();
        progress(BackfillProgress {
            operation: "backfill.start".to_string(),
            tool,
            source_root: tool_root.display().to_string(),
            processed_files: 0,
            total_files,
            source_path: None,
        });
        let processed_files = AtomicUsize::new(0);
        let tool_report = files
            .par_iter()
            .map(|file| {
                // `since` is already filtered upstream by
                // filter_backfill_files_by_since. Here we only guard against a
                // source file that vanishes between enumeration and read: skip
                // it (fail-open) instead of aborting the whole backfill.
                let outcome = backfill_source_file(home, tool, file, &parse_context)
                    .map(source_backfill_report_to_backfill_report);
                let vanished = matches!(&outcome, Err(error) if is_vanished_source(error));
                let result = if vanished {
                    Ok(empty_backfill_report())
                } else {
                    outcome
                };
                let processed = processed_files.fetch_add(1, Ordering::SeqCst) + 1;
                progress(BackfillProgress {
                    operation: if vanished {
                        "backfill.skip_missing".to_string()
                    } else {
                        "backfill.file".to_string()
                    },
                    tool,
                    source_root: tool_root.display().to_string(),
                    processed_files: processed,
                    total_files,
                    source_path: Some(file.display().to_string()),
                });
                result
            })
            .try_reduce(empty_backfill_report, |mut left, right| {
                merge_backfill_report(&mut left, right);
                Ok(left)
            })?;
        merge_backfill_report(&mut report, tool_report);
    }

    detect_deleted_sources(home, source_root, &mut report)?;

    Ok(report)
}

pub fn backfill_dry_run(
    home: &Path,
    selection: Option<Tool>,
    source_root: &Path,
    since: Option<&str>,
) -> Result<BackfillDryRunReport> {
    backfill_dry_run_with_progress(home, selection, source_root, since, |_| {})
}

pub fn backfill_dry_run_with_progress<F>(
    home: &Path,
    selection: Option<Tool>,
    source_root: &Path,
    since: Option<&str>,
    progress: F,
) -> Result<BackfillDryRunReport>
where
    F: Fn(BackfillProgress) + Sync,
{
    init_home(home)?;
    let since_threshold = since.map(parse_since_threshold).transpose()?;
    let tools: Vec<Tool> = match selection {
        Some(tool) => vec![tool],
        None => Tool::all().to_vec(),
    };
    let mut sessions = Vec::new();

    for tool in tools {
        let tool_root = backfill_tool_root(source_root, tool);
        if !tool_root.exists() {
            continue;
        }
        let mut files = Vec::new();
        collect_backfill_files(tool, &tool_root, &mut files)?;
        files.sort();
        let parse_context = backfill_parse_context(tool, &tool_root, &files)?;
        let files = filter_backfill_files_by_since(files, since_threshold)?;
        let total_files = files.len();
        progress(BackfillProgress {
            operation: "backfill.dry_run.start".to_string(),
            tool,
            source_root: tool_root.display().to_string(),
            processed_files: 0,
            total_files,
            source_path: None,
        });
        let processed_files = AtomicUsize::new(0);
        let file_reports: Vec<Result<Vec<BackfillCoverageSession>>> = files
            .par_iter()
            .map(|file| {
                // `since` filtered upstream; tolerate a file that vanishes
                // between enumeration and read (fail-open skip).
                let outcome = backfill_dry_run_file(home, tool, file, &parse_context);
                let vanished = matches!(&outcome, Err(error) if is_vanished_source(error));
                let result = if vanished { Ok(Vec::new()) } else { outcome };
                let processed = processed_files.fetch_add(1, Ordering::SeqCst) + 1;
                progress(BackfillProgress {
                    operation: if vanished {
                        "backfill.dry_run.skip_missing".to_string()
                    } else {
                        "backfill.dry_run.file".to_string()
                    },
                    tool,
                    source_root: tool_root.display().to_string(),
                    processed_files: processed,
                    total_files,
                    source_path: Some(file.display().to_string()),
                });
                result
            })
            .collect();
        for file_report in file_reports {
            sessions.extend(file_report?);
        }
    }

    let source_files = sessions
        .iter()
        .map(|session| session.source_path.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    let on_disk_events = sessions.iter().map(|session| session.on_disk).sum();
    let captured_events = sessions.iter().map(|session| session.captured).sum();
    let missing_events = sessions.iter().map(|session| session.missing).sum();
    let partial_sessions = sessions.iter().filter(|session| session.partial).count();

    Ok(BackfillDryRunReport {
        source_files,
        on_disk_events,
        captured_events,
        missing_events,
        partial_sessions,
        sessions,
    })
}

fn empty_backfill_report() -> BackfillReport {
    BackfillReport {
        source_files: 0,
        appended_events: 0,
        checkpoint_files: 0,
        discontinuities: 0,
    }
}

fn source_backfill_report_to_backfill_report(report: SourceBackfillReport) -> BackfillReport {
    BackfillReport {
        source_files: 1,
        appended_events: report.appended_events,
        checkpoint_files: 1,
        discontinuities: report.discontinuities,
    }
}

fn merge_backfill_report(left: &mut BackfillReport, right: BackfillReport) {
    left.source_files = left.source_files.saturating_add(right.source_files);
    left.appended_events = left.appended_events.saturating_add(right.appended_events);
    left.checkpoint_files = left.checkpoint_files.saturating_add(right.checkpoint_files);
    left.discontinuities = left.discontinuities.saturating_add(right.discontinuities);
}

fn filter_backfill_files_by_since(
    files: Vec<PathBuf>,
    since_threshold: Option<SystemTime>,
) -> Result<Vec<PathBuf>> {
    if since_threshold.is_none() {
        return Ok(files);
    }
    files
        .into_iter()
        .filter_map(
            |file| match should_skip_source_file(&file, since_threshold) {
                Ok(true) => None,
                Ok(false) => Some(Ok(file)),
                // A file that vanishes between enumeration and this mtime stat
                // is skipped, not fatal — same fail-open guarantee the read
                // path gives for the no-`since` case.
                Err(error) if is_vanished_source(&error) => None,
                Err(error) => Some(Err(error)),
            },
        )
        .collect()
}

fn backfill_dry_run_file(
    home: &Path,
    tool: Tool,
    file: &Path,
    parse_context: &BackfillParseContext,
) -> Result<Vec<BackfillCoverageSession>> {
    let parsed = parse_backfill_source(tool, file, 0, parse_context)?;
    let mut by_session: BTreeMap<String, Vec<EventEnvelope>> = BTreeMap::new();
    for event in parsed.events {
        by_session
            .entry(event.session_id.clone())
            .or_default()
            .push(event);
    }
    let mut sessions = Vec::new();
    for (session_id, events) in by_session {
        let (captured, would_import) =
            captured_count_and_import_preview(home, tool, &session_id, &events)?;
        let on_disk = events.len();
        let missing = on_disk.saturating_sub(captured);
        sessions.push(BackfillCoverageSession {
            tool,
            session_id,
            source_path: file.display().to_string(),
            on_disk,
            captured,
            missing,
            partial: missing > 0,
            would_import,
        });
    }
    Ok(sessions)
}

fn backfill_parse_context(
    tool: Tool,
    tool_root: &Path,
    files: &[PathBuf],
) -> Result<BackfillParseContext> {
    if tool != Tool::Opencode {
        return Ok(BackfillParseContext::default());
    }

    let message_session_ids = opencode_message_session_ids(tool_root)?;
    let mut context = OpenCodeBackfillContext::default();

    for file in files
        .iter()
        .filter(|file| opencode_storage_kind(file) == Some("session"))
    {
        let content = fs::read_to_string(file).map_err(|source| Error::Io {
            path: file.to_path_buf(),
            source,
        })?;
        let Ok(payload) = serde_json::from_str::<Value>(&content) else {
            continue;
        };
        let Some(session_id) = opencode_metadata_session_id(&payload, &message_session_ids) else {
            continue;
        };
        context
            .metadata_session_ids
            .insert(file.display().to_string(), session_id.clone());
        if let Some(worktree) = opencode_worktree_for_payload(&payload) {
            context
                .worktree_by_session_id
                .entry(session_id)
                .or_insert(worktree);
        }
    }

    Ok(BackfillParseContext {
        opencode: Some(context),
    })
}

fn opencode_message_session_ids(tool_root: &Path) -> Result<BTreeSet<String>> {
    let message_root = tool_root.join("storage").join("message");
    let mut session_ids = BTreeSet::new();
    if !message_root.exists() {
        return Ok(session_ids);
    }
    for entry in fs::read_dir(&message_root).map_err(|source| Error::Io {
        path: message_root.clone(),
        source,
    })? {
        let entry = entry.map_err(|source| Error::Io {
            path: message_root.clone(),
            source,
        })?;
        if !entry.path().is_dir() {
            continue;
        }
        if let Some(name) = entry.file_name().to_str().filter(|value| !value.is_empty()) {
            session_ids.insert(name.to_string());
        }
    }
    Ok(session_ids)
}

fn opencode_metadata_session_id(
    payload: &Value,
    message_session_ids: &BTreeSet<String>,
) -> Option<String> {
    opencode_direct_session_id(payload)
        .filter(|session_id| {
            message_session_ids.is_empty() || message_session_ids.contains(session_id)
        })
        .or_else(|| {
            let id = string_pointer(payload, "/id")?;
            message_session_ids.contains(&id).then_some(id)
        })
        .or_else(|| {
            if message_session_ids.len() == 1 {
                message_session_ids.iter().next().cloned()
            } else {
                None
            }
        })
}

/// Resolve the session id from a live OpenCode plugin event.
///
/// OpenCode events have no top-level `session_id` (the field the generic hook
/// path requires), so every event the plugin forwards would otherwise fail
/// validation and the ingest subprocess would exit non-zero. Message, part,
/// tool, text, patch, step, and file events carry `sessionID`; the payload may
/// be the bare object or wrapped under `info`/`part`/`properties`. `session.*`
/// events have no `sessionID` and instead identify the session by the session
/// object's own `id` — so that fallback is gated on the event name to avoid
/// mistaking a message id for a session id.
fn opencode_hook_session_id(payload: &Value, event_name: &str) -> Result<String> {
    const SESSION_ID_POINTERS: [&str; 9] = [
        "/sessionID",
        "/session_id",
        "/sessionId",
        "/info/sessionID",
        "/part/sessionID",
        "/properties/info/sessionID",
        "/properties/part/sessionID",
        "/payload/sessionID",
        "/payload/session_id",
    ];
    const SESSION_OBJECT_ID_POINTERS: [&str; 3] = ["/id", "/info/id", "/properties/info/id"];

    SESSION_ID_POINTERS
        .into_iter()
        .find_map(|pointer| string_pointer(payload, pointer))
        .or_else(|| {
            // Only `session.*` events name the session by its own `id`.
            event_name
                .starts_with("session.")
                .then(|| {
                    SESSION_OBJECT_ID_POINTERS
                        .into_iter()
                        .find_map(|pointer| string_pointer(payload, pointer))
                })
                .flatten()
        })
        .ok_or_else(|| {
            Error::Validation(format!(
                "opencode event '{event_name}' has no resolvable session id \
                 (looked for sessionID, and id for session.* events)"
            ))
        })
}

fn opencode_direct_session_id(payload: &Value) -> Option<String> {
    string_pointer(payload, "/session_id")
        .or_else(|| string_pointer(payload, "/sessionID"))
        .or_else(|| string_pointer(payload, "/sessionId"))
        .or_else(|| string_pointer(payload, "/message/sessionID"))
        .or_else(|| string_pointer(payload, "/payload/session_id"))
        .or_else(|| string_pointer(payload, "/payload/sessionID"))
        .or_else(|| string_pointer(payload, "/payload/sessionId"))
        .or_else(|| string_pointer(payload, "/session/id"))
}

fn opencode_storage_kind(path: &Path) -> Option<&'static str> {
    let mut saw_storage = false;
    for component in path.components() {
        let value = component.as_os_str();
        if saw_storage {
            if value == "message" {
                return Some("message");
            }
            if value == "part" {
                return Some("part");
            }
            if value == "session" {
                return Some("session");
            }
            saw_storage = false;
        }
        if value == "storage" {
            saw_storage = true;
        }
    }
    None
}

fn should_skip_source_file(path: &Path, since_threshold: Option<SystemTime>) -> Result<bool> {
    let Some(since_threshold) = since_threshold else {
        return Ok(false);
    };
    let modified = fs::metadata(path)
        .map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?
        .modified()
        .map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
    Ok(modified < since_threshold)
}

fn parse_since_threshold(value: &str) -> Result<SystemTime> {
    parse_date_or_duration_threshold(value, "since")
}

fn normalize_date_or_duration(value: &str, field_name: &str) -> Result<String> {
    let timestamp = parse_date_or_duration_threshold(value, field_name)?;
    Ok(OffsetDateTime::from(timestamp).format(&Rfc3339)?)
}

fn parse_date_or_duration_threshold(value: &str, field_name: &str) -> Result<SystemTime> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(Error::Validation(format!("{field_name} must not be empty")));
    }
    if trimmed.len() > 1 {
        let (number, suffix) = trimmed.split_at(trimmed.len() - 1);
        if matches!(suffix, "d" | "h" | "m" | "s") && number.chars().all(|ch| ch.is_ascii_digit()) {
            let amount = number.parse::<u64>().map_err(|_| {
                Error::Validation(format!("invalid {field_name} duration: {value}"))
            })?;
            let seconds = match suffix {
                "d" => amount.saturating_mul(86_400),
                "h" => amount.saturating_mul(3_600),
                "m" => amount.saturating_mul(60),
                "s" => amount,
                _ => unreachable!(),
            };
            return SystemTime::now()
                .checked_sub(StdDuration::from_secs(seconds))
                .ok_or_else(|| {
                    Error::Validation(format!("invalid {field_name} duration: {value}"))
                });
        }
    }
    if trimmed.len() == 10 && trimmed.as_bytes()[4] == b'-' && trimmed.as_bytes()[7] == b'-' {
        let year = trimmed[0..4]
            .parse::<i32>()
            .map_err(|_| Error::Validation(format!("invalid {field_name} date: {value}")))?;
        let month = trimmed[5..7]
            .parse::<u8>()
            .map_err(|_| Error::Validation(format!("invalid {field_name} date: {value}")))?;
        let day = trimmed[8..10]
            .parse::<u8>()
            .map_err(|_| Error::Validation(format!("invalid {field_name} date: {value}")))?;
        let month = Month::try_from(month)
            .map_err(|_| Error::Validation(format!("invalid {field_name} date: {value}")))?;
        let date = Date::from_calendar_date(year, month, day)
            .map_err(|_| Error::Validation(format!("invalid {field_name} date: {value}")))?;
        let timestamp = date
            .with_hms(0, 0, 0)
            .map_err(|_| Error::Validation(format!("invalid {field_name} date: {value}")))?
            .assume_utc();
        return Ok(SystemTime::from(timestamp));
    }
    let timestamp = OffsetDateTime::parse(trimmed, &Rfc3339)
        .map_err(|_| Error::Validation(format!("invalid {field_name} value: {value}")))?;
    Ok(SystemTime::from(timestamp))
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SourceBackfillReport {
    appended_events: usize,
    discontinuities: usize,
}

#[derive(Debug, Clone, Default)]
struct BackfillParseContext {
    opencode: Option<OpenCodeBackfillContext>,
}

#[derive(Debug, Clone, Default)]
struct OpenCodeBackfillContext {
    metadata_session_ids: HashMap<String, String>,
    worktree_by_session_id: HashMap<String, String>,
}

fn backfill_source_file(
    home: &Path,
    tool: Tool,
    source_path: &Path,
    parse_context: &BackfillParseContext,
) -> Result<SourceBackfillReport> {
    let source_meta = source_file_metadata(source_path)?;
    let source_len = source_meta.size;
    let source_kind = source_kind_for(tool, source_path).to_string();
    let previous_checkpoint = load_checkpoint(home, tool, &source_kind, source_path)?;
    let mut start_offset = 0u64;
    let mut appended_events = 0usize;
    let mut discontinuities = 0usize;
    let now = OffsetDateTime::now_utc().format(&Rfc3339)?;

    if let Some(previous) = previous_checkpoint.as_ref() {
        if previous.byte_offset > source_len {
            append_discontinuity(
                home,
                tool,
                &previous.session_id,
                "source.truncated",
                source_path,
                previous.byte_offset,
                source_len,
            )?;
            discontinuities += 1;
        } else {
            start_offset = previous.byte_offset;
            if previous.source_identity.as_deref() != source_meta.identity.as_deref()
                || start_offset > 0 && !checkpoint_matches_source(source_path, previous)?
            {
                append_discontinuity(
                    home,
                    tool,
                    &previous.session_id,
                    "source.rotated",
                    source_path,
                    previous.byte_offset,
                    source_len,
                )?;
                discontinuities += 1;
                start_offset = 0;
            }
        }
    }

    if start_offset >= source_len && discontinuities == 0 {
        let checkpoint = SourceCheckpoint {
            source_tool: tool,
            source_kind,
            source_path: source_path.display().to_string(),
            source_identity: source_meta.identity,
            session_id: previous_checkpoint
                .as_ref()
                .map(|checkpoint| checkpoint.session_id.clone())
                .or_else(|| session_id_from_source_path(source_path))
                .unwrap_or_else(|| source_path_fallback_session_id(source_path)),
            byte_offset: source_len,
            source_size: source_len,
            source_mtime: source_meta.mtime,
            last_line_hash: last_line_hash(source_path)?,
            last_successful_import_timestamp: now,
        };
        write_checkpoint(home, &checkpoint)?;
        return Ok(SourceBackfillReport {
            appended_events: 0,
            discontinuities: 0,
        });
    }

    let parsed = parse_backfill_source(tool, source_path, start_offset, parse_context)?;
    appended_events += append_prepared_events(home, parsed.events)?
        .into_iter()
        .filter(|report| report.appended)
        .count();

    let checkpoint = SourceCheckpoint {
        source_tool: tool,
        source_kind,
        source_path: source_path.display().to_string(),
        source_identity: source_meta.identity,
        session_id: parsed
            .last_session_id
            .or_else(|| previous_checkpoint.map(|checkpoint| checkpoint.session_id))
            .or_else(|| session_id_from_source_path(source_path))
            .unwrap_or_else(|| source_path_fallback_session_id(source_path)),
        byte_offset: source_len,
        source_size: source_len,
        source_mtime: source_meta.mtime,
        last_line_hash: last_line_hash(source_path)?,
        last_successful_import_timestamp: now,
    };
    write_checkpoint(home, &checkpoint)?;

    Ok(SourceBackfillReport {
        appended_events,
        discontinuities,
    })
}

#[derive(Debug)]
struct ParsedBackfillSource {
    events: Vec<EventEnvelope>,
    last_session_id: Option<String>,
}

fn parse_backfill_source(
    tool: Tool,
    source_path: &Path,
    start_offset: u64,
    parse_context: &BackfillParseContext,
) -> Result<ParsedBackfillSource> {
    match source_path.extension().and_then(|value| value.to_str()) {
        Some("jsonl") => parse_backfill_jsonl(tool, source_path, start_offset, parse_context),
        Some("json") => parse_backfill_json(tool, source_path, parse_context),
        _ => Ok(ParsedBackfillSource {
            events: Vec::new(),
            last_session_id: None,
        }),
    }
}

fn parse_ingest_file_source(
    tool: Tool,
    source: Source,
    source_path: &Path,
) -> Result<ParsedBackfillSource> {
    match (tool, source) {
        (Tool::Codex, Source::ExecJson) | (Tool::Codex, Source::AppServer) => {
            parse_codex_stream_source(source_path)
        }
        (_, Source::ExecJson | Source::AppServer) => Err(Error::Validation(format!(
            "{} source is only supported for codex ingest",
            source.as_str()
        ))),
        _ => parse_backfill_source(tool, source_path, 0, &BackfillParseContext::default()),
    }
}

fn parse_codex_stream_source(source_path: &Path) -> Result<ParsedBackfillSource> {
    match source_path.extension().and_then(|value| value.to_str()) {
        Some("jsonl") => parse_codex_stream_jsonl(source_path),
        Some("json") => parse_codex_stream_json(source_path),
        _ => Ok(ParsedBackfillSource {
            events: Vec::new(),
            last_session_id: None,
        }),
    }
}

fn parse_codex_stream_jsonl(source_path: &Path) -> Result<ParsedBackfillSource> {
    let file = File::open(source_path).map_err(|source| Error::Io {
        path: source_path.to_path_buf(),
        source,
    })?;
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    let mut offset = 0u64;
    let mut events = Vec::new();
    let mut last_session_id = None;

    loop {
        line.clear();
        let bytes = reader.read_line(&mut line).map_err(|source| Error::Io {
            path: source_path.to_path_buf(),
            source,
        })?;
        if bytes == 0 {
            break;
        }
        let line_start = offset;
        offset += bytes as u64;
        if line.trim().is_empty() {
            continue;
        }
        let payload = match serde_json::from_str(line.trim_end()) {
            Ok(payload) => payload,
            Err(error) => malformed_native_payload(source_path, line_start, line.trim_end(), error),
        };
        let event = envelope_from_codex_stream_payload(
            source_path,
            line_start,
            payload,
            last_session_id.as_deref(),
        )?;
        last_session_id = Some(event.session_id.clone());
        events.push(event);
    }

    Ok(ParsedBackfillSource {
        events,
        last_session_id,
    })
}

fn parse_codex_stream_json(source_path: &Path) -> Result<ParsedBackfillSource> {
    let content = fs::read_to_string(source_path).map_err(|source| Error::Io {
        path: source_path.to_path_buf(),
        source,
    })?;
    let payload: Value = match serde_json::from_str(&content) {
        Ok(payload) => payload,
        Err(error) => {
            let event = envelope_from_codex_stream_payload(
                source_path,
                0,
                malformed_native_payload(source_path, 0, &content, error),
                None,
            )?;
            return Ok(ParsedBackfillSource {
                last_session_id: Some(event.session_id.clone()),
                events: vec![event],
            });
        }
    };
    let records = match payload {
        Value::Array(values) => values,
        Value::Object(mut map) => map
            .remove("events")
            .or_else(|| map.remove("notifications"))
            .and_then(|value| value.as_array().cloned())
            .unwrap_or_else(|| vec![Value::Object(map)]),
        _ => Vec::new(),
    };
    let mut events = Vec::new();
    let mut last_session_id = None;
    for (index, payload) in records.into_iter().enumerate() {
        let event = envelope_from_codex_stream_payload(
            source_path,
            index as u64,
            payload,
            last_session_id.as_deref(),
        )?;
        last_session_id = Some(event.session_id.clone());
        events.push(event);
    }

    Ok(ParsedBackfillSource {
        events,
        last_session_id,
    })
}

fn envelope_from_codex_stream_payload(
    source_path: &Path,
    byte_offset: u64,
    payload: Value,
    previous_session_id: Option<&str>,
) -> Result<EventEnvelope> {
    let source_event_type = codex_stream_event_name(&payload);
    let session_id = codex_stream_session_id(source_path, &payload, previous_session_id);
    let canonical_type = payload
        .get("canonical_type")
        .and_then(Value::as_str)
        .map(CanonicalType::from_str)
        .transpose()?
        .unwrap_or_else(|| canonical_type_for_payload(Tool::Codex, &source_event_type, &payload));
    let sequence =
        sequence_for_payload(Tool::Codex, &source_event_type, &payload, Some(byte_offset));
    let source_event_id =
        source_event_id_for_payload(Tool::Codex, &source_event_type, &payload, sequence);

    Ok(EventEnvelope {
        schema_version: SCHEMA_VERSION,
        captured_at: timestamp_for_payload(&payload)
            .unwrap_or(OffsetDateTime::now_utc().format(&Rfc3339)?),
        tool: Tool::Codex,
        tool_version: tool_version_for_payload(&payload),
        session_id: session_id.clone(),
        filename_session_id: sanitize_session_id(&session_id),
        turn_id: turn_id_for_payload(&payload),
        message_id: message_id_for_payload(&payload),
        project_root: project_root_for_payload(&payload),
        cwd: cwd_for_payload(&payload),
        source: Source::ExecJson,
        source_event_type,
        canonical_type,
        source_event_id,
        dedupe_key: String::new(),
        sequence,
        raw_file: None,
        raw_offset: None,
        payload,
        payload_ref: None,
    })
}

fn parse_backfill_jsonl(
    tool: Tool,
    source_path: &Path,
    start_offset: u64,
    parse_context: &BackfillParseContext,
) -> Result<ParsedBackfillSource> {
    let file = File::open(source_path).map_err(|source| Error::Io {
        path: source_path.to_path_buf(),
        source,
    })?;
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    let mut offset = 0u64;
    let mut events = Vec::new();
    let mut last_session_id = None;

    loop {
        line.clear();
        let bytes = reader.read_line(&mut line).map_err(|source| Error::Io {
            path: source_path.to_path_buf(),
            source,
        })?;
        if bytes == 0 {
            break;
        }
        let line_start = offset;
        offset += bytes as u64;
        if offset <= start_offset || line.trim().is_empty() {
            continue;
        }
        let payload = match serde_json::from_str(line.trim_end()) {
            Ok(payload) => payload,
            Err(error) => malformed_native_payload(source_path, line_start, line.trim_end(), error),
        };
        let event =
            envelope_from_backfill_payload(tool, source_path, line_start, payload, parse_context)?;
        last_session_id = Some(event.session_id.clone());
        events.push(event);
    }

    Ok(ParsedBackfillSource {
        events,
        last_session_id,
    })
}

fn parse_backfill_json(
    tool: Tool,
    source_path: &Path,
    parse_context: &BackfillParseContext,
) -> Result<ParsedBackfillSource> {
    let content = fs::read_to_string(source_path).map_err(|source| Error::Io {
        path: source_path.to_path_buf(),
        source,
    })?;
    let payload: Value = match serde_json::from_str(&content) {
        Ok(payload) => payload,
        Err(error) => {
            let event = envelope_from_backfill_payload(
                tool,
                source_path,
                0,
                malformed_native_payload(source_path, 0, &content, error),
                parse_context,
            )?;
            return Ok(ParsedBackfillSource {
                last_session_id: Some(event.session_id.clone()),
                events: vec![event],
            });
        }
    };
    let mut events = Vec::new();
    let mut last_session_id = None;

    match payload {
        Value::Array(values) => {
            for (index, value) in values.into_iter().enumerate() {
                let event = envelope_from_backfill_payload(
                    tool,
                    source_path,
                    index as u64,
                    value,
                    parse_context,
                )?;
                last_session_id = Some(event.session_id.clone());
                events.push(event);
            }
        }
        Value::Object(mut map) => {
            if let Some(Value::Array(messages)) = map.remove("messages") {
                let inherited_session_id = map
                    .get("session_id")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                for (index, mut message) in messages.into_iter().enumerate() {
                    if let (Some(session_id), Value::Object(message_map)) =
                        (inherited_session_id.as_ref(), &mut message)
                    {
                        message_map
                            .entry("session_id".to_string())
                            .or_insert_with(|| Value::String(session_id.clone()));
                    }
                    let event = envelope_from_backfill_payload(
                        tool,
                        source_path,
                        index as u64,
                        message,
                        parse_context,
                    )?;
                    last_session_id = Some(event.session_id.clone());
                    events.push(event);
                }
            } else {
                let event = envelope_from_backfill_payload(
                    tool,
                    source_path,
                    0,
                    Value::Object(map),
                    parse_context,
                )?;
                last_session_id = Some(event.session_id.clone());
                events.push(event);
            }
        }
        _ => {}
    }

    Ok(ParsedBackfillSource {
        events,
        last_session_id,
    })
}

fn envelope_from_backfill_payload(
    tool: Tool,
    source_path: &Path,
    byte_offset: u64,
    payload: Value,
    parse_context: &BackfillParseContext,
) -> Result<EventEnvelope> {
    let session_id = backfill_session_id(tool, source_path, &payload, parse_context);
    let source_event_type = backfill_event_name_for_payload(tool, source_path, &payload);
    let canonical_type = payload
        .get("canonical_type")
        .and_then(Value::as_str)
        .map(CanonicalType::from_str)
        .transpose()?
        .unwrap_or_else(|| canonical_type_for_backfill_payload(tool, &source_event_type, &payload));
    let sequence = sequence_for_payload(tool, &source_event_type, &payload, Some(byte_offset));
    let source_event_id = source_event_id_for_payload(tool, &source_event_type, &payload, sequence);
    Ok(EventEnvelope {
        schema_version: SCHEMA_VERSION,
        captured_at: timestamp_for_payload(&payload)
            .unwrap_or(OffsetDateTime::now_utc().format(&Rfc3339)?),
        tool,
        tool_version: payload
            .get("tool_version")
            .and_then(Value::as_str)
            .or_else(|| payload.pointer("/version").and_then(Value::as_str))
            .or_else(|| {
                payload
                    .pointer("/payload/cli_version")
                    .and_then(Value::as_str)
            })
            .map(str::to_string),
        session_id: session_id.clone(),
        filename_session_id: sanitize_session_id(&session_id),
        turn_id: string_pointer(&payload, "/turn_id")
            .or_else(|| string_pointer(&payload, "/payload/turn_id"))
            .or_else(|| string_pointer(&payload, "/parentUuid"))
            .or_else(|| string_pointer(&payload, "/parentID")),
        message_id: payload
            .get("message_id")
            .or_else(|| payload.pointer("/payload/message_id"))
            .or_else(|| payload.pointer("/message/id"))
            .or_else(|| payload.pointer("/messageID"))
            .or_else(|| payload.pointer("/uuid"))
            .or_else(|| payload.pointer("/id"))
            .and_then(Value::as_str)
            .map(str::to_string),
        project_root: project_root_for_backfill_payload(tool, &payload, &session_id, parse_context),
        cwd: cwd_for_backfill_payload(tool, &payload, &session_id, parse_context),
        source: Source::Backfill,
        source_event_type,
        canonical_type,
        source_event_id,
        dedupe_key: String::new(),
        sequence,
        raw_file: None,
        raw_offset: None,
        payload,
        payload_ref: None,
    })
}

fn codex_stream_event_name(payload: &Value) -> String {
    string_pointer(payload, "/type")
        .or_else(|| string_pointer(payload, "/method"))
        .or_else(|| string_pointer(payload, "/params/type"))
        .or_else(|| string_pointer(payload, "/payload/type"))
        .unwrap_or_else(|| "codex.unknown".to_string())
}

fn codex_stream_session_id(
    source_path: &Path,
    payload: &Value,
    previous_session_id: Option<&str>,
) -> String {
    string_pointer(payload, "/session_id")
        .or_else(|| string_pointer(payload, "/thread_id"))
        .or_else(|| string_pointer(payload, "/threadId"))
        .or_else(|| string_pointer(payload, "/thread/id"))
        .or_else(|| string_pointer(payload, "/payload/session_id"))
        .or_else(|| string_pointer(payload, "/payload/thread_id"))
        .or_else(|| string_pointer(payload, "/payload/thread/id"))
        .or_else(|| string_pointer(payload, "/params/session_id"))
        .or_else(|| string_pointer(payload, "/params/sessionId"))
        .or_else(|| string_pointer(payload, "/params/thread_id"))
        .or_else(|| string_pointer(payload, "/params/threadId"))
        .or_else(|| string_pointer(payload, "/params/thread/id"))
        .or_else(|| {
            let event_name = codex_stream_event_name(payload);
            if matches!(event_name.as_str(), "thread.started" | "thread/started") {
                string_pointer(payload, "/id").or_else(|| string_pointer(payload, "/params/id"))
            } else {
                None
            }
        })
        .or_else(|| previous_session_id.map(str::to_string))
        .or_else(|| session_id_from_source_path(source_path))
        .unwrap_or_else(|| source_path_fallback_session_id(source_path))
}

fn tool_version_for_payload(payload: &Value) -> Option<String> {
    string_pointer(payload, "/tool_version")
        .or_else(|| string_pointer(payload, "/version"))
        .or_else(|| string_pointer(payload, "/payload/tool_version"))
        .or_else(|| string_pointer(payload, "/payload/cli_version"))
        .or_else(|| string_pointer(payload, "/payload/version"))
        .or_else(|| string_pointer(payload, "/params/tool_version"))
        .or_else(|| string_pointer(payload, "/params/cli_version"))
        .or_else(|| string_pointer(payload, "/params/version"))
}

fn turn_id_for_payload(payload: &Value) -> Option<String> {
    string_pointer(payload, "/turn_id")
        .or_else(|| string_pointer(payload, "/turnId"))
        .or_else(|| string_pointer(payload, "/turn/id"))
        .or_else(|| string_pointer(payload, "/payload/turn_id"))
        .or_else(|| string_pointer(payload, "/payload/turnId"))
        .or_else(|| string_pointer(payload, "/payload/turn/id"))
        .or_else(|| string_pointer(payload, "/params/turn_id"))
        .or_else(|| string_pointer(payload, "/params/turnId"))
        .or_else(|| string_pointer(payload, "/params/turn/id"))
        .or_else(|| string_pointer(payload, "/params/item/turn_id"))
        .or_else(|| string_pointer(payload, "/item/turn_id"))
        .or_else(|| string_pointer(payload, "/parentUuid"))
        .or_else(|| string_pointer(payload, "/parentID"))
}

fn message_id_for_payload(payload: &Value) -> Option<String> {
    string_pointer(payload, "/message_id")
        .or_else(|| string_pointer(payload, "/messageId"))
        .or_else(|| string_pointer(payload, "/message/id"))
        .or_else(|| string_pointer(payload, "/messageID"))
        .or_else(|| string_pointer(payload, "/payload/message_id"))
        .or_else(|| string_pointer(payload, "/payload/messageId"))
        .or_else(|| string_pointer(payload, "/payload/message/id"))
        .or_else(|| string_pointer(payload, "/payload/item/message_id"))
        .or_else(|| string_pointer(payload, "/payload/item/messageId"))
        .or_else(|| string_pointer(payload, "/payload/item/id"))
        .or_else(|| string_pointer(payload, "/params/message_id"))
        .or_else(|| string_pointer(payload, "/params/messageId"))
        .or_else(|| string_pointer(payload, "/params/message/id"))
        .or_else(|| string_pointer(payload, "/params/item/message_id"))
        .or_else(|| string_pointer(payload, "/params/item/messageId"))
        .or_else(|| string_pointer(payload, "/params/item/id"))
        .or_else(|| string_pointer(payload, "/item/message_id"))
        .or_else(|| string_pointer(payload, "/item/messageId"))
        .or_else(|| string_pointer(payload, "/item/id"))
        .or_else(|| string_pointer(payload, "/uuid"))
        .or_else(|| string_pointer(payload, "/id"))
}

fn project_root_for_payload(payload: &Value) -> Option<String> {
    string_pointer(payload, "/project_root")
        .or_else(|| string_pointer(payload, "/payload/project_root"))
        .or_else(|| string_pointer(payload, "/params/project_root"))
        .or_else(|| string_pointer(payload, "/path/root"))
        .or_else(|| string_pointer(payload, "/params/path/root"))
        .or_else(|| opencode_worktree_for_payload(payload))
}

fn cwd_for_payload(payload: &Value) -> Option<String> {
    string_pointer(payload, "/cwd")
        .or_else(|| string_pointer(payload, "/payload/cwd"))
        .or_else(|| string_pointer(payload, "/params/cwd"))
        .or_else(|| string_pointer(payload, "/path/cwd"))
        .or_else(|| string_pointer(payload, "/params/path/cwd"))
        .or_else(|| opencode_worktree_for_payload(payload))
}

fn project_root_for_backfill_payload(
    tool: Tool,
    payload: &Value,
    session_id: &str,
    parse_context: &BackfillParseContext,
) -> Option<String> {
    project_root_for_payload(payload).or_else(|| {
        (tool == Tool::Opencode)
            .then(|| {
                parse_context
                    .opencode
                    .as_ref()?
                    .worktree_by_session_id
                    .get(session_id)
                    .cloned()
            })
            .flatten()
    })
}

fn cwd_for_backfill_payload(
    tool: Tool,
    payload: &Value,
    session_id: &str,
    parse_context: &BackfillParseContext,
) -> Option<String> {
    cwd_for_payload(payload)
        .or_else(|| project_root_for_backfill_payload(tool, payload, session_id, parse_context))
}

fn opencode_worktree_for_payload(payload: &Value) -> Option<String> {
    string_pointer(payload, "/worktree")
        .or_else(|| string_pointer(payload, "/payload/worktree"))
        .or_else(|| string_pointer(payload, "/params/worktree"))
        .or_else(|| string_pointer(payload, "/project/worktree"))
        .or_else(|| string_pointer(payload, "/session/worktree"))
}

fn opencode_server_events_from_payload(
    fallback_session_id: &str,
    payload: Value,
) -> Result<Vec<EventEnvelope>> {
    let messages = opencode_server_messages(payload);
    let mut events = Vec::new();
    for (message_index, message) in messages.into_iter().enumerate() {
        let session_id = opencode_message_session_id(fallback_session_id, &message);
        let parts = opencode_message_parts(&message);
        if parts.is_empty() || opencode_message_has_top_level_text(&message) {
            events.push(opencode_server_message_envelope(
                &session_id,
                "message.updated",
                Some(message_index as i64),
                message.clone(),
            )?);
        }
        for (part_index, part) in parts.into_iter().enumerate() {
            let message_id = message_id_for_payload(&message);
            let mut payload = serde_json::json!({
                "session_id": session_id.clone(),
                "message_id": message_id.clone(),
                "part": part,
                "server_message_id": message_id
            });
            if let Value::Object(map) = &mut payload {
                if let Some(project_root) = project_root_for_payload(&message) {
                    map.insert("project_root".to_string(), Value::String(project_root));
                }
                if let Some(cwd) = cwd_for_payload(&message) {
                    map.insert("cwd".to_string(), Value::String(cwd));
                }
                if let Some(worktree) = opencode_worktree_for_payload(&message) {
                    map.insert("worktree".to_string(), Value::String(worktree));
                }
            }
            events.push(opencode_server_message_envelope(
                &session_id,
                "message.part.updated",
                Some(part_index as i64),
                payload,
            )?);
        }
    }
    Ok(events)
}

fn opencode_server_messages(payload: Value) -> Vec<Value> {
    match payload {
        Value::Array(messages) => messages,
        Value::Object(mut map) => map
            .remove("messages")
            .or_else(|| map.remove("data"))
            .or_else(|| map.remove("result"))
            .and_then(|value| match value {
                Value::Array(values) => Some(values),
                Value::Object(mut object) => object.remove("messages").and_then(|nested| {
                    if let Value::Array(values) = nested {
                        Some(values)
                    } else {
                        None
                    }
                }),
                other => Some(vec![other]),
            })
            .unwrap_or_else(|| vec![Value::Object(map)]),
        other => vec![other],
    }
}

fn opencode_message_parts(message: &Value) -> Vec<Value> {
    for pointer in ["/parts", "/message/parts", "/payload/parts"] {
        if let Some(parts) = message.pointer(pointer).and_then(Value::as_array) {
            return parts.clone();
        }
    }
    Vec::new()
}

fn opencode_message_has_top_level_text(message: &Value) -> bool {
    ["text", "content", "message", "summary"].iter().any(|key| {
        message
            .get(*key)
            .and_then(Value::as_str)
            .is_some_and(|value| !value.trim().is_empty())
    })
}

fn opencode_message_session_id(fallback_session_id: &str, message: &Value) -> String {
    string_pointer(message, "/session_id")
        .or_else(|| string_pointer(message, "/sessionID"))
        .or_else(|| string_pointer(message, "/sessionId"))
        .or_else(|| string_pointer(message, "/message/sessionID"))
        .or_else(|| string_pointer(message, "/payload/session_id"))
        .unwrap_or_else(|| fallback_session_id.to_string())
}

fn opencode_server_message_envelope(
    session_id: &str,
    source_event_type: &str,
    sequence: Option<i64>,
    payload: Value,
) -> Result<EventEnvelope> {
    let canonical_type = canonical_type_for_opencode_native(source_event_type, &payload);
    let source_event_id =
        source_event_id_for_payload(Tool::Opencode, source_event_type, &payload, sequence);
    Ok(EventEnvelope {
        schema_version: SCHEMA_VERSION,
        captured_at: timestamp_for_payload(&payload)
            .unwrap_or(OffsetDateTime::now_utc().format(&Rfc3339)?),
        tool: Tool::Opencode,
        tool_version: tool_version_for_payload(&payload),
        session_id: session_id.to_string(),
        filename_session_id: sanitize_session_id(session_id),
        turn_id: turn_id_for_payload(&payload),
        message_id: message_id_for_payload(&payload),
        project_root: project_root_for_payload(&payload),
        cwd: cwd_for_payload(&payload),
        source: Source::Backfill,
        source_event_type: source_event_type.to_string(),
        canonical_type,
        source_event_id,
        dedupe_key: String::new(),
        sequence,
        raw_file: None,
        raw_offset: None,
        payload,
        payload_ref: None,
    })
}

fn backfill_session_id(
    tool: Tool,
    source_path: &Path,
    payload: &Value,
    parse_context: &BackfillParseContext,
) -> String {
    if tool == Tool::Opencode && opencode_storage_kind(source_path) == Some("session") {
        if let Some(session_id) = parse_context
            .opencode
            .as_ref()
            .and_then(|context| {
                context
                    .metadata_session_ids
                    .get(&source_path.display().to_string())
            })
            .cloned()
            .or_else(|| opencode_direct_session_id(payload))
        {
            return session_id;
        }
    }
    string_pointer(payload, "/session_id")
        .or_else(|| string_pointer(payload, "/payload/session_id"))
        .or_else(|| string_pointer(payload, "/sessionId"))
        .or_else(|| string_pointer(payload, "/sessionID"))
        .or_else(|| codex_session_meta_id(tool, payload))
        .or_else(|| session_id_from_source_path(source_path))
        .unwrap_or_else(|| source_path_fallback_session_id(source_path))
}

fn backfill_event_name_for_payload(tool: Tool, source_path: &Path, payload: &Value) -> String {
    if let Ok(name) = hook_event_name(payload) {
        return name.to_string();
    }
    match tool {
        Tool::Claude => {
            let kind = string_pointer(payload, "/type").unwrap_or_else(|| "unknown".to_string());
            format!("claude.{kind}")
        }
        Tool::Codex => {
            string_pointer(payload, "/type").unwrap_or_else(|| "codex.unknown".to_string())
        }
        Tool::Opencode => {
            if opencode_storage_kind(source_path) == Some("session") {
                "session.created".to_string()
            } else if opencode_storage_kind(source_path) == Some("part") {
                match opencode_part_type(payload).as_deref() {
                    Some("tool") | Some("tool-call") => "tool.execute.before".to_string(),
                    Some("tool-result") | Some("tool_result") => "tool.execute.after".to_string(),
                    Some(kind) => kind.to_string(),
                    None => "message.part.updated".to_string(),
                }
            } else {
                "message.updated".to_string()
            }
        }
    }
}

fn canonical_type_for_backfill_payload(
    tool: Tool,
    source_event_type: &str,
    payload: &Value,
) -> CanonicalType {
    if payload.get("parse_error").is_some() || payload.get("raw_line").is_some() {
        return CanonicalType::Error;
    }
    if payload.get("hook_event_name").is_some() || payload.get("event").is_some() {
        return canonical_type_for_payload(tool, source_event_type, payload);
    }
    match tool {
        Tool::Claude => canonical_type_for_claude_native(payload),
        Tool::Codex => canonical_type_for_payload(tool, source_event_type, payload),
        Tool::Opencode => canonical_type_for_opencode_native(source_event_type, payload),
    }
}

fn canonical_type_for_claude_native(payload: &Value) -> CanonicalType {
    match string_pointer(payload, "/type").as_deref() {
        Some("user") => {
            if claude_message_has_content_type(payload, "tool_result") {
                CanonicalType::ToolResult
            } else {
                CanonicalType::UserMessage
            }
        }
        Some("assistant") => {
            if claude_message_has_content_type(payload, "tool_use")
                && !claude_message_has_text(payload)
            {
                CanonicalType::ToolCall
            } else {
                CanonicalType::AssistantMessage
            }
        }
        Some("summary") => CanonicalType::CompactionAfter,
        Some("attachment") => match string_pointer(payload, "/attachment/hookEvent").as_deref() {
            Some("PreToolUse") => CanonicalType::ToolCall,
            Some("PostToolUse") | Some("PostToolUseFailure") | Some("PostToolBatch") => {
                CanonicalType::ToolResult
            }
            _ => CanonicalType::Error,
        },
        Some("queue-operation") => CanonicalType::SessionResumed,
        Some("system") => CanonicalType::SessionStarted,
        _ => CanonicalType::Error,
    }
}

fn claude_message_has_content_type(payload: &Value, content_type: &str) -> bool {
    payload
        .pointer("/message/content")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .any(|item| item.get("type").and_then(Value::as_str) == Some(content_type))
        })
        .unwrap_or(false)
}

fn claude_message_has_text(payload: &Value) -> bool {
    payload
        .pointer("/message/content")
        .and_then(Value::as_array)
        .map(|items| {
            items.iter().any(|item| {
                item.get("type").and_then(Value::as_str) == Some("text")
                    && item
                        .get("text")
                        .and_then(Value::as_str)
                        .is_some_and(|text| !text.trim().is_empty())
            })
        })
        .unwrap_or(false)
}

fn canonical_type_for_opencode_native(source_event_type: &str, payload: &Value) -> CanonicalType {
    match source_event_type {
        "tool.execute.before" => return CanonicalType::ToolCall,
        "tool.execute.after" => return CanonicalType::ToolResult,
        "session.created" | "session.updated" => return CanonicalType::SessionStarted,
        "reasoning" | "step-start" | "step-finish" => return CanonicalType::AssistantDelta,
        "patch" => return CanonicalType::FileChanged,
        _ => {}
    }
    match opencode_part_type(payload).as_deref() {
        Some("reasoning") | Some("step-start") | Some("step-finish") => {
            return CanonicalType::AssistantDelta;
        }
        Some("patch") => return CanonicalType::FileChanged,
        Some("text") => return CanonicalType::AssistantMessage,
        Some("tool") | Some("tool-call") => return CanonicalType::ToolCall,
        Some("tool-result") | Some("tool_result") => return CanonicalType::ToolResult,
        Some("file") => return CanonicalType::FileChanged,
        _ => {}
    }
    match string_pointer(payload, "/role").as_deref() {
        Some("user") => CanonicalType::UserMessage,
        Some("assistant") => CanonicalType::AssistantMessage,
        _ => CanonicalType::Error,
    }
}

fn opencode_part_type(payload: &Value) -> Option<String> {
    string_pointer(payload, "/part/type")
        .or_else(|| string_pointer(payload, "/payload/part/type"))
        .or_else(|| string_pointer(payload, "/type"))
        .or_else(|| string_pointer(payload, "/payload/type"))
}

fn timestamp_for_payload(payload: &Value) -> Option<String> {
    string_pointer(payload, "/captured_at")
        .or_else(|| string_pointer(payload, "/timestamp"))
        .or_else(|| string_pointer(payload, "/payload/timestamp"))
        .or_else(|| string_pointer(payload, "/params/timestamp"))
        .or_else(|| string_pointer(payload, "/created_at"))
        .or_else(|| string_pointer(payload, "/payload/created_at"))
        .or_else(|| string_pointer(payload, "/params/created_at"))
        .or_else(|| {
            payload
                .pointer("/time/created")
                .and_then(Value::as_i64)
                .and_then(timestamp_millis_to_rfc3339)
        })
        .or_else(|| {
            payload
                .pointer("/time/created")
                .and_then(Value::as_u64)
                .and_then(|value| i64::try_from(value).ok())
                .and_then(timestamp_millis_to_rfc3339)
        })
        .or_else(|| {
            payload
                .pointer("/params/time/created")
                .and_then(Value::as_i64)
                .and_then(timestamp_millis_to_rfc3339)
        })
        .or_else(|| {
            payload
                .pointer("/params/time/created")
                .and_then(Value::as_u64)
                .and_then(|value| i64::try_from(value).ok())
                .and_then(timestamp_millis_to_rfc3339)
        })
}

fn timestamp_millis_to_rfc3339(value: i64) -> Option<String> {
    OffsetDateTime::from_unix_timestamp(value.div_euclid(1000))
        .ok()
        .and_then(|timestamp| timestamp.format(&Rfc3339).ok())
}

fn malformed_native_payload(
    source_path: &Path,
    byte_offset: u64,
    raw_line: &str,
    error: serde_json::Error,
) -> Value {
    serde_json::json!({
        "type": "parse_error",
        "source_path": source_path.display().to_string(),
        "byte_offset": byte_offset,
        "parse_error": error.to_string(),
        "raw_line": raw_line
    })
}

fn codex_session_meta_id(tool: Tool, payload: &Value) -> Option<String> {
    if tool == Tool::Codex && payload.get("type").and_then(Value::as_str) == Some("session_meta") {
        return string_pointer(payload, "/payload/id");
    }
    None
}

fn session_id_from_source_path(source_path: &Path) -> Option<String> {
    let stem = source_path.file_stem()?.to_str()?.trim();
    if stem.is_empty() {
        return None;
    }

    if stem.len() >= 36 {
        let suffix = &stem[stem.len() - 36..];
        if looks_like_uuid(suffix) {
            return Some(suffix.to_string());
        }
    }

    Some(stem.to_string())
}

fn source_path_fallback_session_id(source_path: &Path) -> String {
    let mut hasher = Sha256::new();
    hasher.update(source_path.display().to_string().as_bytes());
    let hash = hex::encode(hasher.finalize());
    format!("source-{}", &hash[..16])
}

fn looks_like_uuid(value: &str) -> bool {
    value.len() == 36
        && value.char_indices().all(|(index, character)| {
            matches!(index, 8 | 13 | 18 | 23) && character == '-'
                || !matches!(index, 8 | 13 | 18 | 23) && character.is_ascii_hexdigit()
        })
}

fn string_pointer(payload: &Value, pointer: &str) -> Option<String> {
    payload
        .pointer(pointer)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn append_prepared_events(home: &Path, events: Vec<EventEnvelope>) -> Result<Vec<AppendReport>> {
    let mut grouped = BTreeMap::<PathBuf, Vec<EventEnvelope>>::new();
    for event in events {
        let raw_file = canonical_raw_path(home, event.tool, &event.session_id);
        grouped.entry(raw_file).or_default().push(event);
    }

    let mut reports = Vec::new();
    for (raw_file, events) in grouped {
        reports.extend(append_prepared_events_to_raw_file(home, &raw_file, events)?);
    }
    Ok(reports)
}

fn captured_count_and_import_preview(
    home: &Path,
    tool: Tool,
    session_id: &str,
    events: &[EventEnvelope],
) -> Result<(usize, Vec<BackfillImportPreview>)> {
    let raw_file = canonical_raw_path(home, tool, session_id);
    let existing = existing_dedupe_events_for_raw_file(home, &raw_file)?;
    let mut captured = 0usize;
    let mut would_import = Vec::new();
    for event in events {
        let key = dedupe_key(DedupeParts {
            tool: event.tool,
            session_id: &event.session_id,
            canonical_type: event.canonical_type,
            source_event_id: event.source_event_id.as_deref(),
            sequence: event.sequence,
            payload: &event.payload,
        })?;
        if existing.contains_key(&key) {
            captured += 1;
        } else {
            would_import.push(BackfillImportPreview {
                canonical_type: event.canonical_type.as_str().to_string(),
                source_event_type: event.source_event_type.clone(),
                source_event_id: event.source_event_id.clone(),
                sequence: event.sequence,
                captured_at: event.captured_at.clone(),
            });
        }
    }
    Ok((captured, would_import))
}

fn existing_dedupe_events_for_raw_file(
    home: &Path,
    raw_file: &Path,
) -> Result<HashMap<String, ExistingRawEvent>> {
    let sidecar = DedupeSidecarFiles::for_raw_file(home, raw_file);
    match load_full_dedupe_sidecar_events(raw_file, &sidecar) {
        Ok(Some(events)) => Ok(events),
        Ok(None) | Err(_) => Ok(read_raw_dedupe_snapshot(raw_file)?.events),
    }
}

fn append_prepared_events_to_raw_file(
    home: &Path,
    raw_file: &Path,
    events: Vec<EventEnvelope>,
) -> Result<Vec<AppendReport>> {
    if events.is_empty() {
        return Ok(Vec::new());
    }
    if let Some(parent) = raw_file.parent() {
        create_dir_0700(parent)?;
    }
    let lock_path = lock_path_for_raw_file(raw_file);
    let lock_file = OpenOptions::new()
        .create(true)
        // Lock sentinel: content is never written, so do not truncate.
        .truncate(false)
        .read(true)
        .write(true)
        .open(&lock_path)
        .map_err(|source| Error::Io {
            path: lock_path.clone(),
            source,
        })?;
    chmod(&lock_path, 0o600)?;
    lock_file.lock_exclusive().map_err(|source| Error::Io {
        path: lock_path.clone(),
        source,
    })?;

    let append_result = append_envelopes_locked(home, raw_file, events);
    let unlock_result = FileExt::unlock(&lock_file).map_err(|source| Error::Io {
        path: lock_path,
        source,
    });

    match (append_result, unlock_result) {
        (Ok(report), Ok(())) => Ok(report),
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error),
    }
}

fn append_prepared_event(home: &Path, event: EventEnvelope) -> Result<AppendReport> {
    let raw_file = canonical_raw_path(home, event.tool, &event.session_id);
    if let Some(parent) = raw_file.parent() {
        create_dir_0700(parent)?;
    }
    let lock_path = lock_path_for_raw_file(&raw_file);
    let lock_file = OpenOptions::new()
        .create(true)
        // Lock sentinel: content is never written, so do not truncate.
        .truncate(false)
        .read(true)
        .write(true)
        .open(&lock_path)
        .map_err(|source| Error::Io {
            path: lock_path.clone(),
            source,
        })?;
    chmod(&lock_path, 0o600)?;
    lock_file.lock_exclusive().map_err(|source| Error::Io {
        path: lock_path.clone(),
        source,
    })?;

    let append_result = append_envelope_locked(home, &raw_file, event);
    let unlock_result = FileExt::unlock(&lock_file).map_err(|source| Error::Io {
        path: lock_path,
        source,
    });

    match (append_result, unlock_result) {
        (Ok(report), Ok(())) => Ok(report),
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error),
    }
}

fn append_discontinuity(
    home: &Path,
    tool: Tool,
    session_id: &str,
    source_event_type: &str,
    source_path: &Path,
    previous_offset: u64,
    current_len: u64,
) -> Result<()> {
    let payload = serde_json::json!({
        "session_id": session_id,
        "hook_event_name": source_event_type,
        "source_path": source_path.display().to_string(),
        "previous_byte_offset": previous_offset,
        "current_byte_len": current_len,
        "reason": source_event_type,
        "text": "backfill source discontinuity"
    });
    append_prepared_event(
        home,
        EventEnvelope {
            schema_version: SCHEMA_VERSION,
            captured_at: OffsetDateTime::now_utc().format(&Rfc3339)?,
            tool,
            tool_version: None,
            session_id: session_id.to_string(),
            filename_session_id: sanitize_session_id(session_id),
            turn_id: None,
            message_id: None,
            project_root: None,
            cwd: None,
            source: Source::Backfill,
            source_event_type: source_event_type.to_string(),
            canonical_type: CanonicalType::SourceDiscontinuity,
            source_event_id: Some(format!(
                "{}:{}:{}:{}",
                source_event_type,
                source_path.display(),
                previous_offset,
                current_len
            )),
            dedupe_key: String::new(),
            sequence: None,
            raw_file: None,
            raw_offset: None,
            payload,
            payload_ref: None,
        },
    )?;
    Ok(())
}

fn backfill_tool_root(source_root: &Path, tool: Tool) -> PathBuf {
    let candidate = match tool {
        Tool::Codex => source_root.join("codex"),
        Tool::Claude => source_root.join("claude-code"),
        Tool::Opencode => source_root.join("opencode"),
    };
    if candidate.exists() {
        candidate
    } else {
        source_root.to_path_buf()
    }
}

fn collect_backfill_files(tool: Tool, dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
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
        if path.is_dir() {
            collect_backfill_files(tool, &path, files)?;
            continue;
        }
        if is_backfill_candidate(tool, &path) {
            files.push(path);
        }
    }
    Ok(())
}

fn is_backfill_candidate(tool: Tool, path: &Path) -> bool {
    match tool {
        Tool::Claude => is_claude_transcript_file(path),
        Tool::Codex | Tool::Opencode => matches!(
            path.extension().and_then(|value| value.to_str()),
            Some("json") | Some("jsonl")
        ),
    }
}

fn is_claude_transcript_file(path: &Path) -> bool {
    if path.extension().and_then(|value| value.to_str()) != Some("jsonl") {
        return false;
    }
    let Some(stem) = path.file_stem().and_then(|value| value.to_str()) else {
        return false;
    };
    looks_like_uuid(stem) || stem == "transcript" || stem.starts_with("agent-")
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SourceFileMetadata {
    identity: Option<String>,
    size: u64,
    mtime: Option<i64>,
}

fn source_file_metadata(path: &Path) -> Result<SourceFileMetadata> {
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
fn source_file_identity(_path: &Path, metadata: &fs::Metadata) -> Option<String> {
    use std::os::unix::fs::MetadataExt;

    Some(format!("{}:{}", metadata.dev(), metadata.ino()))
}

#[cfg(not(unix))]
fn source_file_identity(path: &Path, _metadata: &fs::Metadata) -> Option<String> {
    fs::canonicalize(path)
        .ok()
        .map(|path| path.display().to_string())
}

fn system_time_to_unix_seconds(value: SystemTime) -> Option<i64> {
    value
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| i64::try_from(duration.as_secs()).ok())
}

fn load_checkpoint(
    home: &Path,
    tool: Tool,
    source_kind: &str,
    source_path: &Path,
) -> Result<Option<SourceCheckpoint>> {
    let db_path = home.join("index").join("harness.db");
    let conn = open_index(&db_path)?;
    load_checkpoint_from_conn(&conn, &db_path, tool, source_kind, source_path)
}

fn load_checkpoint_from_conn(
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

fn write_checkpoint(home: &Path, checkpoint: &SourceCheckpoint) -> Result<()> {
    let db_path = home.join("index").join("harness.db");
    let conn = open_index(&db_path)?;
    write_checkpoint_to_conn(&conn, &db_path, checkpoint)
}

fn write_checkpoint_to_conn(
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

fn last_line_hash(path: &Path) -> Result<Option<String>> {
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

fn checkpoint_matches_source(path: &Path, checkpoint: &SourceCheckpoint) -> Result<bool> {
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

fn raw_index_checkpoint_is_current(
    conn: &Connection,
    db_path: &Path,
    tool: Tool,
    source_path: &Path,
    source_meta: &SourceFileMetadata,
) -> Result<bool> {
    let Some(checkpoint) =
        load_checkpoint_from_conn(conn, db_path, tool, "raw_jsonl", source_path)?
    else {
        return Ok(false);
    };

    Ok(
        checkpoint.source_identity.as_deref() == source_meta.identity.as_deref()
            && checkpoint.byte_offset == source_meta.size
            && checkpoint.source_size == source_meta.size
            && checkpoint.source_mtime == source_meta.mtime,
    )
}

fn write_raw_index_checkpoint(
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

fn raw_index_checkpoint_session_id(tool: Tool, source_path: &Path) -> String {
    let Some(stem) = source_path.file_stem().and_then(|value| value.to_str()) else {
        return source_path_fallback_session_id(source_path);
    };
    let prefix = format!("{}_", tool.as_str());
    stem.strip_prefix(&prefix)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| source_path_fallback_session_id(source_path))
}

fn hash_line(line: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(line.as_bytes());
    hex::encode(hasher.finalize())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

fn source_kind_for(tool: Tool, source_path: &Path) -> &'static str {
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

fn detect_deleted_sources(
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

fn checkpoints_under_root(home: &Path, source_root: &Path) -> Result<Vec<SourceCheckpoint>> {
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

fn delete_checkpoint(home: &Path, checkpoint: &SourceCheckpoint) -> Result<()> {
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

fn table_count(conn: &Connection, db_path: &Path, table: &str) -> Result<i64> {
    conn.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
        row.get(0)
    })
    .map_err(|source| Error::Sqlite {
        path: db_path.to_path_buf(),
        source,
    })
}

fn table_exists(conn: &Connection, db_path: &Path, table: &str) -> Result<bool> {
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

fn storage_footprint(home: &Path) -> StorageFootprint {
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

fn directory_size(path: &Path) -> Result<u64> {
    if !path.exists() {
        return Ok(0);
    }
    let metadata = fs::metadata(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    if metadata.is_file() {
        return Ok(metadata.len());
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
        total = total.saturating_add(directory_size(&entry.path())?);
    }
    Ok(total)
}

fn create_dir_0700(path: &Path) -> Result<()> {
    fs::create_dir_all(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    chmod(path, 0o700)
}

fn create_config_if_missing(path: &Path) -> Result<()> {
    match OpenOptions::new().write(true).create_new(true).open(path) {
        Ok(mut file) => {
            file.write_all(
                b"schema_version = 1\n\n[opencode]\n# server_url = \"http://127.0.0.1:4096\"\n",
            )
            .map_err(|source| Error::Io {
                path: path.to_path_buf(),
                source,
            })?;
        }
        Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(source) => {
            return Err(Error::Io {
                path: path.to_path_buf(),
                source,
            });
        }
    }
    chmod(path, 0o600)
}

fn read_opencode_server_url_from_config(path: &Path) -> Result<Option<String>> {
    if !path.exists() {
        return Ok(None);
    }
    let content = fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut in_opencode_section = false;
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            in_opencode_section = line == "[opencode]";
            continue;
        }
        if !in_opencode_section {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        if key.trim() != "server_url" {
            continue;
        }
        let value = value
            .trim()
            .trim_matches('"')
            .trim_matches('\'')
            .trim()
            .to_string();
        if value.is_empty() {
            return Ok(None);
        }
        return Ok(Some(value));
    }
    Ok(None)
}

/// Set (`Some`) or clear (`None`) the `[opencode] server_url` in the user's
/// `config.toml`, preserving every other line, key, and comment byte-for-byte.
/// Writes atomically (temp file + rename, 0o600) and is idempotent — when the
/// resulting content is unchanged it does not touch the file. This is the only
/// config write path the wizard uses; it must never reset unrelated settings.
pub fn set_opencode_server_url(home: &Path, url: Option<&str>) -> Result<()> {
    let path = home.join("config.toml");
    create_config_if_missing(&path)?;
    let content = fs::read_to_string(&path).map_err(|source| Error::Io {
        path: path.clone(),
        source,
    })?;
    let updated = rewrite_opencode_server_url(&content, url);
    if updated == content {
        return Ok(());
    }
    let tmp = home.join("config.toml.tmp");
    fs::write(&tmp, &updated).map_err(|source| Error::Io {
        path: tmp.clone(),
        source,
    })?;
    chmod(&tmp, 0o600)?;
    fs::rename(&tmp, &path).map_err(|source| Error::Io {
        path: path.clone(),
        source,
    })?;
    Ok(())
}

/// Pure rewrite used by [`set_opencode_server_url`]. Replaces or inserts the
/// `server_url` line inside `[opencode]` when `url` is `Some`, or removes an
/// active (uncommented) `server_url` line when `None`; all other content is
/// preserved verbatim.
fn rewrite_opencode_server_url(content: &str, url: Option<&str>) -> String {
    let new_line = url.map(|value| format!("server_url = \"{value}\""));
    let mut out: Vec<String> = Vec::new();
    let mut in_opencode = false;
    let mut handled = false;

    for raw in content.lines() {
        let trimmed = raw.trim();
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            // Leaving the [opencode] section: a pending set is inserted here; a
            // clear with nothing to remove is simply marked handled.
            if in_opencode && !handled {
                if let Some(line) = &new_line {
                    out.push(line.clone());
                }
                handled = true;
            }
            in_opencode = trimmed == "[opencode]";
            out.push(raw.to_string());
            continue;
        }
        if in_opencode && !handled {
            let is_comment = trimmed.starts_with('#');
            let body = trimmed.trim_start_matches('#').trim();
            let is_server_url = body
                .split_once('=')
                .map(|(key, _)| key.trim() == "server_url")
                .unwrap_or(false);
            if is_server_url {
                match (&new_line, is_comment) {
                    // Set: replace this line (whether it was the commented seed
                    // hint or a live value).
                    (Some(line), _) => {
                        out.push(line.clone());
                        handled = true;
                        continue;
                    }
                    // Clear: drop the active line.
                    (None, false) => {
                        handled = true;
                        continue;
                    }
                    // Clear: leave a commented hint untouched; keep scanning.
                    (None, true) => {}
                }
            }
        }
        out.push(raw.to_string());
    }

    // EOF still inside [opencode]: append a pending set under the section.
    if in_opencode && !handled {
        if let Some(line) = &new_line {
            out.push(line.clone());
        }
        handled = true;
    }
    // No [opencode] section at all: append one (only when setting).
    if !handled {
        if let Some(line) = &new_line {
            if out.last().is_some_and(|last| !last.is_empty()) {
                out.push(String::new());
            }
            out.push("[opencode]".to_string());
            out.push(line.clone());
        }
    }

    let mut result = out.join("\n");
    if content.ends_with('\n') {
        result.push('\n');
    }
    result
}

fn initialize_database(path: &Path) -> Result<()> {
    register_semantic_extension_if_enabled();
    let conn = Connection::open(path).map_err(|source| Error::Sqlite {
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
    ensure_events_fts_schema(&conn, path)?;
    ensure_supporting_indexes(&conn, path)?;
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

fn open_index(path: &Path) -> Result<Connection> {
    register_semantic_extension_if_enabled();
    let conn = Connection::open(path).map_err(|source| Error::Sqlite {
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
    ensure_events_fts_schema(&conn, path)?;
    ensure_supporting_indexes(&conn, path)?;
    Ok(conn)
}

fn register_semantic_extension_if_enabled() {
    #[cfg(feature = "semantic")]
    {
        static SQLITE_VEC_REGISTER: std::sync::Once = std::sync::Once::new();
        SQLITE_VEC_REGISTER.call_once(|| unsafe {
            rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
                sqlite_vec::sqlite3_vec_init as *const (),
            )));
        });
    }
}

#[cfg(feature = "semantic")]
fn ensure_semantic_vector_schema(conn: &Connection, path: &Path) -> Result<()> {
    conn.execute_batch(SEMANTIC_VECTOR_SCHEMA)
        .map_err(|source| Error::Sqlite {
            path: path.to_path_buf(),
            source,
        })?;
    Ok(())
}

#[cfg(not(feature = "semantic"))]
fn ensure_semantic_vector_schema(_conn: &Connection, _path: &Path) -> Result<()> {
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

fn ensure_table_column(
    conn: &Connection,
    path: &Path,
    table: &str,
    column: &str,
    definition: &str,
) -> Result<()> {
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

fn ensure_events_fts_schema(conn: &Connection, path: &Path) -> Result<()> {
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

    if columns.iter().any(|column| column == "searchable_text") || !contentless {
        conn.execute_batch("DROP TABLE IF EXISTS events_fts;")
            .map_err(|source| Error::Sqlite {
                path: path.to_path_buf(),
                source,
            })?;
        conn.execute_batch(EVENTS_FTS_SCHEMA)
            .map_err(|source| Error::Sqlite {
                path: path.to_path_buf(),
                source,
            })?;
        rebuild_events_fts(conn, path)?;
    }

    Ok(())
}

fn ensure_supporting_indexes(conn: &Connection, path: &Path) -> Result<()> {
    conn.execute_batch(
        "CREATE INDEX IF NOT EXISTS idx_events_tool_captured ON events(tool, captured_at);",
    )
    .map_err(|source| Error::Sqlite {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(())
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct RawIndexFileReport {
    indexed_events: usize,
    bytes_read: u64,
    last_line_hash: Option<String>,
}

fn index_raw_file(conn: &Connection, tool: Tool, path: &Path) -> Result<RawIndexFileReport> {
    let file = File::open(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    let mut raw_line = 0i64;
    let mut raw_offset = 0u64;
    let mut indexed = 0usize;
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
        raw_line += 1;
        last_hash = Some(hash_line(line.trim_end()));
        let parsed: EventEnvelope = serde_json::from_str(line.trim_end())?;
        parsed.validate()?;
        if parsed.tool != tool {
            return Err(Error::Validation(format!(
                "raw file {} contains tool {}",
                path.display(),
                parsed.tool
            )));
        }

        if insert_indexed_event(conn, path, raw_line, raw_offset as i64, &parsed)? {
            indexed += 1;
        }
        raw_offset += bytes as u64;
    }

    Ok(RawIndexFileReport {
        indexed_events: indexed,
        bytes_read: raw_offset,
        last_line_hash: last_hash,
    })
}

fn insert_indexed_event(
    conn: &Connection,
    path: &Path,
    raw_line: i64,
    fallback_raw_offset: i64,
    envelope: &EventEnvelope,
) -> Result<bool> {
    let payload = resolved_payload_for_envelope(path, envelope)?;
    let search_document = search_document_for_event(envelope.canonical_type, &payload);
    let searchable_text = search_document.render();
    let raw_file = path.display().to_string();
    let raw_offset = envelope.raw_offset.unwrap_or(fallback_raw_offset);
    let payload_json: Option<String> = None;

    conn.execute(
        "INSERT INTO sessions(
           tool,
           session_id,
           filename_session_id,
           project_root,
           cwd,
           started_at,
           updated_at,
           raw_file
         )
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?6, ?7)
         ON CONFLICT(tool, session_id) DO UPDATE SET
           started_at = CASE
             WHEN sessions.started_at IS NULL OR sessions.started_at > excluded.started_at
             THEN excluded.started_at
             ELSE sessions.started_at
           END,
           updated_at = CASE
             WHEN sessions.updated_at IS NULL OR sessions.updated_at < excluded.updated_at
             THEN excluded.updated_at
             ELSE sessions.updated_at
           END,
           project_root = COALESCE(sessions.project_root, excluded.project_root),
           cwd = COALESCE(sessions.cwd, excluded.cwd)",
        (
            envelope.tool.as_str(),
            &envelope.session_id,
            &envelope.filename_session_id,
            envelope.project_root.as_deref(),
            envelope.cwd.as_deref(),
            &envelope.captured_at,
            &raw_file,
        ),
    )
    .map_err(|source| Error::Sqlite {
        path: path.to_path_buf(),
        source,
    })?;

    let inserted =
        conn.execute(
            "INSERT OR IGNORE INTO events(
              tool,
              session_id,
              dedupe_key,
              schema_version,
              captured_at,
              tool_version,
              turn_id,
              message_id,
              project_root,
              cwd,
              source,
              source_event_type,
              source_event_id,
              canonical_type,
              sequence,
              raw_file,
              raw_line,
              raw_offset,
              payload_json,
              payload_ref,
              searchable_text,
              compaction_state
            )
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15,
                    ?16, ?17, ?18, ?19, ?20, ?21, ?22)",
            params![
                envelope.tool.as_str(),
                &envelope.session_id,
                &envelope.dedupe_key,
                envelope.schema_version as i64,
                &envelope.captured_at,
                envelope.tool_version.as_deref(),
                envelope.turn_id.as_deref(),
                envelope.message_id.as_deref(),
                envelope.project_root.as_deref(),
                envelope.cwd.as_deref(),
                envelope.source.as_str(),
                &envelope.source_event_type,
                envelope.source_event_id.as_deref(),
                envelope.canonical_type.as_str(),
                envelope.sequence,
                &raw_file,
                raw_line,
                raw_offset,
                payload_json.as_deref(),
                envelope.payload_ref.as_deref(),
                &searchable_text,
                compaction_state_for(envelope.canonical_type),
            ],
        )
        .map_err(|source| Error::Sqlite {
            path: path.to_path_buf(),
            source,
        })? == 1;

    if !inserted {
        return Ok(false);
    }

    let event_id = conn.last_insert_rowid();
    insert_derived_rows(
        conn,
        path,
        event_id,
        envelope,
        &payload,
        raw_line,
        raw_offset,
        &search_document,
    )?;
    Ok(true)
}

// Row-insert helper threading the parsed event plus its raw-file coordinates;
// the arguments are distinct positional facts, not a struct in disguise.
#[allow(clippy::too_many_arguments)]
fn insert_derived_rows(
    conn: &Connection,
    path: &Path,
    event_id: i64,
    envelope: &EventEnvelope,
    payload: &Value,
    raw_line: i64,
    raw_offset: i64,
    search_document: &SearchDocument,
) -> Result<()> {
    if let Some(role) = role_for(envelope.canonical_type) {
        conn.execute(
            "INSERT INTO messages(event_id, tool, session_id, role, text, is_delta, sequence)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            (
                event_id,
                envelope.tool.as_str(),
                &envelope.session_id,
                role,
                message_text_for_document(envelope.canonical_type, search_document),
                i64::from(envelope.canonical_type == CanonicalType::AssistantDelta),
                envelope.sequence,
            ),
        )
        .map_err(|source| Error::Sqlite {
            path: path.to_path_buf(),
            source,
        })?;
    }

    if matches!(
        envelope.canonical_type,
        CanonicalType::ToolCall | CanonicalType::ToolResult
    ) {
        conn.execute(
            "INSERT INTO tool_events(event_id, tool, session_id, tool_name, command, status, input_text, output_text)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            (
                event_id,
                envelope.tool.as_str(),
                &envelope.session_id,
                string_field(payload, "tool_name"),
                string_field(payload, "command"),
                tool_status_for(envelope.canonical_type),
                string_field(payload, "input"),
                string_field(payload, "output"),
            ),
        )
        .map_err(|source| Error::Sqlite {
            path: path.to_path_buf(),
            source,
        })?;
    }

    insert_event_file_rows(conn, path, event_id, envelope, payload)?;

    if matches!(
        envelope.canonical_type,
        CanonicalType::CompactionBefore | CanonicalType::CompactionAfter
    ) {
        conn.execute(
            "INSERT INTO compactions(event_id, tool, session_id, trigger, raw_file, raw_line, raw_offset, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            (
                event_id,
                envelope.tool.as_str(),
                &envelope.session_id,
                string_field(payload, "trigger"),
                path.display().to_string(),
                raw_line,
                raw_offset,
                &envelope.captured_at,
            ),
        )
        .map_err(|source| Error::Sqlite {
            path: path.to_path_buf(),
            source,
        })?;
    }

    conn.execute(
        "INSERT INTO events_fts(rowid, user_text, assistant_text, tool_intent, tool_output, metadata_text, tool, session_id, canonical_type, raw_file, raw_line, raw_offset)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
        (
            event_id,
            &search_document.user_text,
            &search_document.assistant_text,
            &search_document.tool_intent,
            &search_document.tool_output,
            &search_document.metadata_text,
            envelope.tool.as_str(),
            &envelope.session_id,
            envelope.canonical_type.as_str(),
            path.display().to_string(),
            raw_line,
            raw_offset,
        ),
    )
    .map_err(|source| Error::Sqlite {
        path: path.to_path_buf(),
        source,
    })?;

    insert_vector_unit_rows(
        conn,
        path,
        event_id,
        envelope,
        raw_line,
        raw_offset,
        search_document,
    )?;

    Ok(())
}

#[cfg(feature = "semantic")]
fn insert_vector_unit_rows(
    conn: &Connection,
    path: &Path,
    event_id: i64,
    envelope: &EventEnvelope,
    raw_line: i64,
    raw_offset: i64,
    search_document: &SearchDocument,
) -> Result<()> {
    let created_at = OffsetDateTime::now_utc().format(&Rfc3339)?;
    for unit in embedding_units_for_document(search_document) {
        insert_vector_unit_text(conn, path, &unit, &created_at)?;
        conn.execute(
            "INSERT OR IGNORE INTO vector_units(
               event_id,
               tool,
               session_id,
               unit_kind,
               unit_index,
               text_hash,
               raw_file,
               raw_line,
               raw_offset,
               created_at
             )
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                event_id,
                envelope.tool.as_str(),
                &envelope.session_id,
                unit.kind.as_str(),
                unit.unit_index as i64,
                &unit.text_hash,
                path.display().to_string(),
                raw_line,
                raw_offset,
                &created_at,
            ],
        )
        .map_err(|source| Error::Sqlite {
            path: path.to_path_buf(),
            source,
        })?;
    }
    Ok(())
}

#[cfg(feature = "semantic")]
fn insert_vector_unit_text(
    conn: &Connection,
    path: &Path,
    unit: &EmbeddingUnit,
    created_at: &str,
) -> Result<()> {
    conn.execute(
        "INSERT OR IGNORE INTO vector_unit_texts(text_hash, text, created_at)
         VALUES (?1, ?2, ?3)",
        params![&unit.text_hash, &unit.text, created_at],
    )
    .map_err(|source| Error::Sqlite {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(())
}

#[cfg(not(feature = "semantic"))]
fn insert_vector_unit_rows(
    _conn: &Connection,
    _path: &Path,
    _event_id: i64,
    _envelope: &EventEnvelope,
    _raw_line: i64,
    _raw_offset: i64,
    _search_document: &SearchDocument,
) -> Result<()> {
    Ok(())
}

#[cfg(feature = "semantic")]
fn embed_index_if_available_with_progress<F>(home: &Path, mut progress: F) -> Result<usize>
where
    F: FnMut(EmbeddingIndexProgress),
{
    if !semantic_model_files_present(home) {
        return Ok(0);
    }
    let db_path = home.join("index").join("harness.db");
    let mut conn = open_index(&db_path)?;
    ensure_semantic_vector_schema(&conn, &db_path)?;
    sync_vector_units(&conn, &db_path)?;
    let total_units = count_unembedded_units(&conn, &db_path)?;
    if total_units == 0 {
        return Ok(0);
    }

    let planned_threads = semantic_intra_threads();
    progress(embedding_index_plan_progress(
        total_units,
        SEMANTIC_EMBED_BATCH_SIZE,
        SEMANTIC_EMBED_WRITE_CHUNK_SIZE,
        planned_threads,
    ));
    progress(embedding_index_phase_progress(
        "loading_model",
        "started",
        SEMANTIC_EMBED_BATCH_SIZE,
        SEMANTIC_EMBED_WRITE_CHUNK_SIZE,
        planned_threads,
    ));
    let Some(embedder) = load_local_embedder(home)? else {
        return Ok(0);
    };
    progress(embedding_index_phase_progress(
        "loading_model",
        "completed",
        embedder.document_batch_size(),
        SEMANTIC_EMBED_WRITE_CHUNK_SIZE,
        embedder.intra_threads(),
    ));
    embed_unembedded_units_paged_with_config(
        &mut conn,
        &db_path,
        &*embedder,
        total_units,
        EmbeddingWriteConfig::default(),
        progress,
    )
}

#[cfg(not(feature = "semantic"))]
fn embed_index_if_available_with_progress<F>(_home: &Path, _progress: F) -> Result<usize>
where
    F: FnMut(EmbeddingIndexProgress),
{
    Ok(0)
}

#[cfg(feature = "semantic")]
fn sync_vector_units(conn: &Connection, db_path: &Path) -> Result<usize> {
    let mut statement = conn
        .prepare(
            "SELECT id, payload_json, tool, session_id, canonical_type, raw_file, raw_line, raw_offset
             FROM events
             WHERE NOT EXISTS (
               SELECT 1 FROM vector_units vu WHERE vu.event_id = events.id
             )
             ORDER BY id",
        )
        .map_err(|source| Error::Sqlite {
            path: db_path.to_path_buf(),
            source,
        })?;
    let rows = statement
        .query_map([], |row| {
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
            path: db_path.to_path_buf(),
            source,
        })?;

    let mut inserted = 0usize;
    for row in rows {
        let (
            event_id,
            payload_json,
            tool,
            session_id,
            canonical_type,
            raw_file,
            raw_line,
            raw_offset,
        ) = row.map_err(|source| Error::Sqlite {
            path: db_path.to_path_buf(),
            source,
        })?;
        let canonical_type = CanonicalType::from_str(&canonical_type)?;
        let payload = match payload_json.as_deref() {
            Some(payload_json) => serde_json::from_str(payload_json)?,
            None => payload_for_raw_pointer(&raw_file, raw_line, raw_offset)?,
        };
        let document = search_document_for_event(canonical_type, &payload);
        let created_at = OffsetDateTime::now_utc().format(&Rfc3339)?;
        for unit in embedding_units_for_document(&document) {
            insert_vector_unit_text(conn, db_path, &unit, &created_at)?;
            let changed = conn
                .execute(
                    "INSERT OR IGNORE INTO vector_units(
                       event_id,
                       tool,
                       session_id,
                       unit_kind,
                       unit_index,
                       text_hash,
                       raw_file,
                       raw_line,
                       raw_offset,
                       created_at
                     )
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                    params![
                        event_id,
                        &tool,
                        &session_id,
                        unit.kind.as_str(),
                        unit.unit_index as i64,
                        &unit.text_hash,
                        &raw_file,
                        raw_line,
                        raw_offset,
                        &created_at,
                    ],
                )
                .map_err(|source| Error::Sqlite {
                    path: db_path.to_path_buf(),
                    source,
                })?;
            inserted = inserted.saturating_add(changed);
        }
    }
    Ok(inserted)
}

#[cfg(feature = "semantic")]
fn embed_unembedded_units_paged_with_config(
    conn: &mut Connection,
    db_path: &Path,
    embedder: &dyn Embedder,
    total_units: usize,
    config: EmbeddingWriteConfig,
    mut progress: impl FnMut(EmbeddingIndexProgress),
) -> Result<usize> {
    let mut embedded = 0usize;
    let mut pending_writes = Vec::with_capacity(embedder.document_batch_size());
    let started = Instant::now();
    let mut last_emit = started;
    let mut after_unit_id = 0i64;

    progress(embedding_index_progress(
        "embedding",
        "started",
        embedded,
        total_units,
        started,
        embedder,
        config.write_chunk_size,
    ));

    loop {
        let mut page = collect_unembedded_units_page(
            conn,
            db_path,
            after_unit_id,
            SEMANTIC_EMBED_COLLECT_BATCH_SIZE,
        )?;
        if page.rows_seen == 0 {
            break;
        }
        after_unit_id = page.last_unit_id;
        bucket_unembedded_units(&mut page.units);

        for batch in page.units.chunks(embedder.document_batch_size()) {
            let texts = batch
                .iter()
                .map(|unit| unit.text.clone())
                .collect::<Vec<_>>();
            let vectors = embedder.embed_documents(&texts)?;
            for (unit, vector) in batch.iter().zip(vectors) {
                pending_writes.push((unit.unit_id, vector));
                embedded += 1;
                if pending_writes.len() >= config.write_chunk_size {
                    flush_embedding_writes(conn, db_path, &pending_writes)?;
                    pending_writes.clear();
                }
            }
            if embedded < total_units && last_emit.elapsed() >= SEMANTIC_EMBED_PROGRESS_INTERVAL {
                progress(embedding_index_progress(
                    "embedding",
                    "running",
                    embedded,
                    total_units,
                    started,
                    embedder,
                    config.write_chunk_size,
                ));
                last_emit = Instant::now();
            }
        }
    }

    if !pending_writes.is_empty() {
        flush_embedding_writes(conn, db_path, &pending_writes)?;
    }
    progress(embedding_index_progress(
        "embedding",
        "completed",
        embedded,
        total_units,
        started,
        embedder,
        config.write_chunk_size,
    ));
    Ok(embedded)
}

#[cfg(feature = "semantic")]
#[cfg_attr(not(test), allow(dead_code))]
fn embed_unembedded_units_with_config(
    conn: &mut Connection,
    db_path: &Path,
    embedder: &dyn Embedder,
    config: EmbeddingWriteConfig,
    progress: impl FnMut(EmbeddingIndexProgress),
) -> Result<usize> {
    let units = collect_unembedded_units(conn, db_path)?;
    embed_collected_unembedded_units_with_config(conn, db_path, embedder, units, config, progress)
}

#[cfg(feature = "semantic")]
fn embed_collected_unembedded_units_with_config(
    conn: &mut Connection,
    db_path: &Path,
    embedder: &dyn Embedder,
    mut units: Vec<UnembeddedUnit>,
    config: EmbeddingWriteConfig,
    mut progress: impl FnMut(EmbeddingIndexProgress),
) -> Result<usize> {
    bucket_unembedded_units(&mut units);
    let total_units = units.len();
    let mut embedded = 0usize;
    let mut pending_writes = Vec::with_capacity(embedder.document_batch_size());
    let started = Instant::now();
    let mut last_emit = started;
    progress(embedding_index_progress(
        "embedding",
        "started",
        embedded,
        total_units,
        started,
        embedder,
        config.write_chunk_size,
    ));
    for batch in units.chunks(embedder.document_batch_size()) {
        let texts = batch
            .iter()
            .map(|unit| unit.text.clone())
            .collect::<Vec<_>>();
        let vectors = embedder.embed_documents(&texts)?;
        for (unit, vector) in batch.iter().zip(vectors) {
            pending_writes.push((unit.unit_id, vector));
            embedded += 1;
            if pending_writes.len() >= config.write_chunk_size {
                flush_embedding_writes(conn, db_path, &pending_writes)?;
                pending_writes.clear();
            }
        }
        if embedded < total_units && last_emit.elapsed() >= SEMANTIC_EMBED_PROGRESS_INTERVAL {
            progress(embedding_index_progress(
                "embedding",
                "running",
                embedded,
                total_units,
                started,
                embedder,
                config.write_chunk_size,
            ));
            last_emit = Instant::now();
        }
    }
    if !pending_writes.is_empty() {
        flush_embedding_writes(conn, db_path, &pending_writes)?;
    }
    progress(embedding_index_progress(
        "embedding",
        "completed",
        embedded,
        total_units,
        started,
        embedder,
        config.write_chunk_size,
    ));
    Ok(embedded)
}

#[cfg(feature = "semantic")]
#[derive(Debug, Clone, Copy)]
struct EmbeddingWriteConfig {
    write_chunk_size: usize,
}

#[cfg(feature = "semantic")]
impl Default for EmbeddingWriteConfig {
    fn default() -> Self {
        Self {
            write_chunk_size: SEMANTIC_EMBED_WRITE_CHUNK_SIZE,
        }
    }
}

#[cfg(feature = "semantic")]
fn flush_embedding_writes(
    conn: &mut Connection,
    db_path: &Path,
    rows: &[(i64, Vec<f32>)],
) -> Result<()> {
    let tx = conn.transaction().map_err(|source| Error::Sqlite {
        path: db_path.to_path_buf(),
        source,
    })?;
    {
        let mut statement = tx
            .prepare(
                "INSERT OR REPLACE INTO vector_unit_embeddings(unit_id, embedding)
                 VALUES (?1, ?2)",
            )
            .map_err(|source| Error::Sqlite {
                path: db_path.to_path_buf(),
                source,
            })?;
        for (unit_id, vector) in rows {
            statement
                .execute(params![unit_id, vector_to_blob(vector)?])
                .map_err(|source| Error::Sqlite {
                    path: db_path.to_path_buf(),
                    source,
                })?;
        }
    }
    tx.commit().map_err(|source| Error::Sqlite {
        path: db_path.to_path_buf(),
        source,
    })?;
    Ok(())
}

#[cfg(feature = "semantic")]
fn embedding_index_progress(
    phase: &str,
    status: &str,
    embedded_units: usize,
    total_units: usize,
    started: Instant,
    embedder: &dyn Embedder,
    write_chunk_size: usize,
) -> EmbeddingIndexProgress {
    let elapsed_seconds = started.elapsed().as_secs_f64();
    let units_per_second = if embedded_units == 0 || elapsed_seconds <= f64::EPSILON {
        0.0
    } else {
        embedded_units as f64 / elapsed_seconds
    };
    let eta_seconds = if units_per_second > 0.0 && embedded_units < total_units {
        Some(((total_units - embedded_units) as f64 / units_per_second).ceil() as u64)
    } else {
        None
    };
    EmbeddingIndexProgress {
        phase: phase.to_string(),
        status: status.to_string(),
        embedded_units,
        total_units,
        units_per_second,
        eta_seconds,
        batch_size: embedder.document_batch_size(),
        write_chunk_size,
        intra_threads: embedder.intra_threads(),
    }
}

#[cfg(feature = "semantic")]
fn embedding_index_phase_progress(
    phase: &str,
    status: &str,
    batch_size: usize,
    write_chunk_size: usize,
    intra_threads: usize,
) -> EmbeddingIndexProgress {
    EmbeddingIndexProgress {
        phase: phase.to_string(),
        status: status.to_string(),
        embedded_units: 0,
        total_units: 0,
        units_per_second: 0.0,
        eta_seconds: None,
        batch_size,
        write_chunk_size,
        intra_threads,
    }
}

#[cfg(feature = "semantic")]
fn embedding_index_plan_progress(
    total_units: usize,
    batch_size: usize,
    write_chunk_size: usize,
    intra_threads: usize,
) -> EmbeddingIndexProgress {
    EmbeddingIndexProgress {
        phase: "embedding_plan".to_string(),
        status: "ready".to_string(),
        embedded_units: 0,
        total_units,
        units_per_second: 0.0,
        eta_seconds: None,
        batch_size,
        write_chunk_size,
        intra_threads,
    }
}

#[cfg(feature = "semantic")]
#[derive(Debug, Clone)]
struct UnembeddedUnit {
    unit_id: i64,
    text: String,
    estimated_tokens: usize,
}

#[cfg(feature = "semantic")]
struct UnembeddedUnitPage {
    units: Vec<UnembeddedUnit>,
    last_unit_id: i64,
    rows_seen: usize,
}

#[cfg(feature = "semantic")]
fn bucket_unembedded_units(units: &mut [UnembeddedUnit]) {
    units.sort_by_key(|unit| (embedding_length_bucket(unit.estimated_tokens), unit.unit_id));
}

#[cfg(feature = "semantic")]
fn embedding_length_bucket(tokens: usize) -> usize {
    match tokens {
        0..=64 => 64,
        65..=128 => 128,
        129..=256 => 256,
        257..=512 => 512,
        513..=1024 => 1024,
        _ => SEMANTIC_EMBED_MAX_LENGTH,
    }
}

#[cfg(feature = "semantic")]
fn estimated_embedding_token_count(text: &str) -> usize {
    let by_words = text.split_whitespace().count();
    let by_chars = text.chars().count().div_ceil(4);
    by_words.max(by_chars).min(SEMANTIC_EMBED_MAX_LENGTH)
}

#[cfg(feature = "semantic")]
fn count_unembedded_units(conn: &Connection, db_path: &Path) -> Result<usize> {
    let count = conn
        .query_row(
            "SELECT COUNT(*)
             FROM vector_units vu
             LEFT JOIN vector_unit_embeddings ve ON ve.unit_id = vu.id
             WHERE ve.unit_id IS NULL",
            [],
            |row| row.get::<_, i64>(0),
        )
        .map_err(|source| Error::Sqlite {
            path: db_path.to_path_buf(),
            source,
        })?;
    Ok(count.max(0) as usize)
}

#[cfg(feature = "semantic")]
fn collect_unembedded_units(conn: &Connection, db_path: &Path) -> Result<Vec<UnembeddedUnit>> {
    let mut units = Vec::new();
    let mut after_unit_id = 0i64;
    loop {
        let page = collect_unembedded_units_page(
            conn,
            db_path,
            after_unit_id,
            SEMANTIC_EMBED_COLLECT_BATCH_SIZE,
        )?;
        if page.rows_seen == 0 {
            break;
        }
        after_unit_id = page.last_unit_id;
        units.extend(page.units);
    }
    Ok(units)
}

#[cfg(feature = "semantic")]
fn collect_unembedded_units_page(
    conn: &Connection,
    db_path: &Path,
    after_unit_id: i64,
    limit: usize,
) -> Result<UnembeddedUnitPage> {
    let mut statement = conn
        .prepare(
            "SELECT
               vu.id,
               vu.unit_kind,
               vu.unit_index,
               vu.text_hash,
               vut.text,
               e.canonical_type,
               e.payload_json,
               e.raw_file,
               e.raw_line,
               e.raw_offset
             FROM vector_units vu
             JOIN events e ON e.id = vu.event_id
             LEFT JOIN vector_unit_texts vut ON vut.text_hash = vu.text_hash
             LEFT JOIN vector_unit_embeddings ve ON ve.unit_id = vu.id
             WHERE ve.unit_id IS NULL
               AND vu.id > ?1
             ORDER BY vu.id
             LIMIT ?2",
        )
        .map_err(|source| Error::Sqlite {
            path: db_path.to_path_buf(),
            source,
        })?;
    let mut rows = statement
        .query(params![
            after_unit_id,
            limit.clamp(1, SEMANTIC_EMBED_COLLECT_BATCH_SIZE) as i64
        ])
        .map_err(|source| Error::Sqlite {
            path: db_path.to_path_buf(),
            source,
        })?;

    let mut units = Vec::new();
    let mut rows_seen = 0usize;
    let mut last_unit_id = after_unit_id;
    while let Some(row) = rows.next().map_err(|source| Error::Sqlite {
        path: db_path.to_path_buf(),
        source,
    })? {
        rows_seen += 1;
        let unit_id = row.get::<_, i64>(0).map_err(|source| Error::Sqlite {
            path: db_path.to_path_buf(),
            source,
        })?;
        last_unit_id = unit_id;
        let unit_kind = row.get::<_, String>(1).map_err(|source| Error::Sqlite {
            path: db_path.to_path_buf(),
            source,
        })?;
        let unit_index = row.get::<_, i64>(2).map_err(|source| Error::Sqlite {
            path: db_path.to_path_buf(),
            source,
        })?;
        let text_hash = row.get::<_, String>(3).map_err(|source| Error::Sqlite {
            path: db_path.to_path_buf(),
            source,
        })?;
        let stored_text = row
            .get::<_, Option<String>>(4)
            .map_err(|source| Error::Sqlite {
                path: db_path.to_path_buf(),
                source,
            })?;
        let canonical_type = row.get::<_, String>(5).map_err(|source| Error::Sqlite {
            path: db_path.to_path_buf(),
            source,
        })?;
        let payload_json = row
            .get::<_, Option<String>>(6)
            .map_err(|source| Error::Sqlite {
                path: db_path.to_path_buf(),
                source,
            })?;
        let raw_file = row.get::<_, String>(7).map_err(|source| Error::Sqlite {
            path: db_path.to_path_buf(),
            source,
        })?;
        let raw_line = row.get::<_, i64>(8).map_err(|source| Error::Sqlite {
            path: db_path.to_path_buf(),
            source,
        })?;
        let raw_offset = row
            .get::<_, Option<i64>>(9)
            .map_err(|source| Error::Sqlite {
                path: db_path.to_path_buf(),
                source,
            })?;

        if let Some(text) = stored_text {
            units.push(UnembeddedUnit {
                unit_id,
                estimated_tokens: estimated_embedding_token_count(&text),
                text,
            });
            continue;
        }

        let canonical_type = CanonicalType::from_str(&canonical_type)?;
        let payload = match payload_json.as_deref() {
            Some(payload_json) => serde_json::from_str(payload_json)?,
            None => payload_for_raw_pointer(&raw_file, raw_line, raw_offset)?,
        };
        let unit_kind = EmbeddingUnitKind::from_str(&unit_kind)?;
        let unit_index = usize::try_from(unit_index)
            .map_err(|_| Error::Validation(format!("negative vector unit index: {unit_index}")))?;
        let document = search_document_for_event(canonical_type, &payload);
        if let Some(unit) = embedding_units_for_document(&document)
            .into_iter()
            .find(|unit| {
                unit.kind == unit_kind
                    && unit.unit_index == unit_index
                    && unit.text_hash == text_hash
            })
        {
            let created_at = OffsetDateTime::now_utc().format(&Rfc3339)?;
            insert_vector_unit_text(conn, db_path, &unit, &created_at)?;
            units.push(UnembeddedUnit {
                unit_id,
                estimated_tokens: estimated_embedding_token_count(&unit.text),
                text: unit.text,
            });
        }
    }
    Ok(UnembeddedUnitPage {
        units,
        last_unit_id,
        rows_seen,
    })
}

#[cfg(feature = "semantic")]
fn vector_to_blob(vector: &[f32]) -> Result<Vec<u8>> {
    if vector.len() != SEMANTIC_VECTOR_DIMENSIONS {
        return Err(Error::SemanticUnavailable(format!(
            "vector has {} dimensions, expected {}",
            vector.len(),
            SEMANTIC_VECTOR_DIMENSIONS
        )));
    }
    let mut blob = Vec::with_capacity(vector.len() * std::mem::size_of::<f32>());
    for value in vector {
        blob.extend_from_slice(&value.to_le_bytes());
    }
    Ok(blob)
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

fn file_paths_for_payload(payload: &Value) -> Vec<String> {
    let mut paths = BTreeSet::new();
    collect_file_paths(payload, None, &mut paths);
    paths.into_iter().collect()
}

fn collect_file_paths(value: &Value, key: Option<&str>, output: &mut BTreeSet<String>) {
    match value {
        Value::String(text) => {
            if key.is_some_and(is_file_path_key) || looks_like_file_path(text) {
                let text = text.trim();
                if !text.is_empty() {
                    output.insert(text.to_string());
                }
            }
        }
        Value::Array(values) => {
            for value in values {
                collect_file_paths(value, key, output);
            }
        }
        Value::Object(map) => {
            for (key, value) in map {
                collect_file_paths(value, Some(key), output);
            }
        }
        _ => {}
    }
}

fn is_file_path_key(key: &str) -> bool {
    matches!(
        key,
        "file" | "file_path" | "filepath" | "path" | "source_path" | "transcript_path"
    ) || key.ends_with("_file")
        || key.ends_with("_path")
}

fn looks_like_file_path(value: &str) -> bool {
    let value = value.trim();
    value.starts_with('/')
        || value.starts_with("./")
        || value.starts_with("../")
        || value.starts_with("~/")
}

fn recalculate_session_counts(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "UPDATE sessions
         SET event_count = (
           SELECT COUNT(*) FROM events
           WHERE events.tool = sessions.tool AND events.session_id = sessions.session_id
         ),
         message_count = (
           SELECT COUNT(*) FROM messages
           WHERE messages.tool = sessions.tool AND messages.session_id = sessions.session_id
         ),
         tool_event_count = (
           SELECT COUNT(*) FROM tool_events
           WHERE tool_events.tool = sessions.tool AND tool_events.session_id = sessions.session_id
         ),
         compaction_count = (
           SELECT COUNT(*) FROM compactions
           WHERE compactions.tool = sessions.tool AND compactions.session_id = sessions.session_id
         );",
    )
    .map_err(|source| Error::Sqlite {
        path: PathBuf::from("harness.db"),
        source,
    })
}

fn required_string<'a>(payload: &'a Value, key: &str) -> Result<&'a str> {
    payload
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| Error::Validation(format!("payload.{key} must be a non-empty string")))
}

fn hook_event_name(payload: &Value) -> Result<&str> {
    payload
        .get("hook_event_name")
        .or_else(|| payload.get("event"))
        .or_else(|| payload.get("type"))
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            Error::Validation(
                "payload.hook_event_name, payload.event, or payload.type is required".to_string(),
            )
        })
}

fn canonical_type_for_payload(
    tool: Tool,
    source_event_type: &str,
    payload: &Value,
) -> CanonicalType {
    if source_event_type == "MessageDisplay" {
        return if payload
            .get("final")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            CanonicalType::AssistantMessage
        } else {
            CanonicalType::AssistantDelta
        };
    }

    if tool == Tool::Codex {
        match source_event_type {
            "session_meta" => return CanonicalType::SessionStarted,
            "turn_context" => return CanonicalType::SessionResumed,
            "response_item" => {
                return match string_pointer(payload, "/payload/type").as_deref() {
                    Some("message") => match string_pointer(payload, "/payload/role").as_deref() {
                        Some("user") => CanonicalType::UserMessage,
                        Some("assistant") => CanonicalType::AssistantMessage,
                        _ => CanonicalType::Error,
                    },
                    Some("function_call") | Some("custom_tool_call") => CanonicalType::ToolCall,
                    Some("function_call_output") | Some("custom_tool_call_output") => {
                        CanonicalType::ToolResult
                    }
                    Some("reasoning") => CanonicalType::AssistantDelta,
                    _ => CanonicalType::Error,
                };
            }
            "event_msg" => {
                return match string_pointer(payload, "/payload/type").as_deref() {
                    Some("user_message") => CanonicalType::UserMessage,
                    Some("agent_message") => CanonicalType::AssistantMessage,
                    Some("agent_reasoning") => CanonicalType::AssistantDelta,
                    Some("exec_command_begin") | Some("tool_call") => CanonicalType::ToolCall,
                    Some("exec_command_end") | Some("tool_output") => CanonicalType::ToolResult,
                    _ => CanonicalType::Error,
                };
            }
            _ => {}
        }
    }

    match (tool, source_event_type) {
        (_, "SessionStart") | (Tool::Opencode, "session.created") => CanonicalType::SessionStarted,
        (Tool::Codex, "thread.started")
        | (Tool::Codex, "thread/started")
        | (Tool::Codex, "turn.started")
        | (Tool::Codex, "turn/started") => CanonicalType::SessionStarted,
        (_, "SessionEnd") | (_, "Stop") | (Tool::Opencode, "session.idle") => {
            CanonicalType::SessionEnded
        }
        (Tool::Codex, "turn.completed") | (Tool::Codex, "turn/completed") => {
            CanonicalType::SessionEnded
        }
        (_, "UserPromptSubmit") => CanonicalType::UserMessage,
        (Tool::Codex, "item/agentMessage/delta") => CanonicalType::AssistantDelta,
        (Tool::Codex, "item.completed") | (Tool::Codex, "item/completed") => {
            canonical_type_for_codex_item(payload).unwrap_or(CanonicalType::AssistantMessage)
        }
        (Tool::Opencode, "message.part.updated") => CanonicalType::AssistantDelta,
        (Tool::Opencode, "message.part.removed") => CanonicalType::AssistantDelta,
        (Tool::Opencode, "message.updated") => CanonicalType::AssistantMessage,
        (Tool::Opencode, "message.removed") => CanonicalType::AssistantMessage,
        (_, "PreToolUse")
        | (Tool::Codex, "SubagentStart")
        | (Tool::Codex, "item.started")
        | (Tool::Codex, "item/started")
        | (Tool::Opencode, "tool.execute.before") => CanonicalType::ToolCall,
        (_, "PostToolUse")
        | (_, "PostToolUseFailure")
        | (_, "PostToolBatch")
        | (Tool::Codex, "SubagentStop")
        | (Tool::Codex, "item.failed")
        | (Tool::Codex, "item/failed")
        | (Tool::Opencode, "tool.execute.after")
        | (Tool::Opencode, "command.executed") => CanonicalType::ToolResult,
        (_, "PreCompact") => CanonicalType::CompactionBefore,
        (_, "PostCompact") | (Tool::Opencode, "session.compacted") => {
            CanonicalType::CompactionAfter
        }
        (Tool::Opencode, "session.updated") => CanonicalType::SessionResumed,
        (Tool::Opencode, "file.edited") => CanonicalType::FileChanged,
        (Tool::Opencode, "session.error") => CanonicalType::Error,
        _ => CanonicalType::Error,
    }
}

fn canonical_type_for_codex_item(payload: &Value) -> Option<CanonicalType> {
    let item = payload
        .pointer("/item")
        .or_else(|| payload.pointer("/payload/item"))
        .or_else(|| payload.pointer("/params/item"))
        .or_else(|| payload.pointer("/payload"))
        .or_else(|| payload.pointer("/params"))
        .unwrap_or(payload);

    match string_pointer(item, "/role").as_deref() {
        Some("user") => return Some(CanonicalType::UserMessage),
        Some("assistant") => return Some(CanonicalType::AssistantMessage),
        _ => {}
    }

    match string_pointer(item, "/type").as_deref() {
        Some("message") | Some("agent_message") | Some("agentMessage") => {
            Some(CanonicalType::AssistantMessage)
        }
        Some("user_message") | Some("userMessage") => Some(CanonicalType::UserMessage),
        Some("function_call")
        | Some("custom_tool_call")
        | Some("tool_call")
        | Some("tool")
        | Some("exec_command_begin") => Some(CanonicalType::ToolCall),
        Some("function_call_output")
        | Some("custom_tool_call_output")
        | Some("tool_result")
        | Some("tool-output")
        | Some("exec_command_end") => Some(CanonicalType::ToolResult),
        Some("reasoning") | Some("agent_reasoning") => Some(CanonicalType::AssistantDelta),
        Some("error") => Some(CanonicalType::Error),
        _ => None,
    }
}

fn source_event_id_for_payload(
    tool: Tool,
    source_event_type: &str,
    payload: &Value,
    sequence: Option<i64>,
) -> Option<String> {
    if source_event_type == "MessageDisplay" {
        if let Some(message_id) = payload.get("message_id").and_then(Value::as_str) {
            let index = payload
                .get("index")
                .and_then(Value::as_i64)
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".to_string());
            let final_state = payload
                .get("final")
                .and_then(Value::as_bool)
                .map(|value| value.to_string())
                .unwrap_or_else(|| "false".to_string());
            return Some(format!("{message_id}:{index}:{final_state}"));
        }
    }
    if source_event_type == "item/agentMessage/delta" {
        if let Some(message_id) = message_id_for_payload(payload) {
            let sequence = sequence
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".to_string());
            return Some(format!("{message_id}:{sequence}:delta"));
        }
    }
    if tool == Tool::Opencode
        && matches!(
            source_event_type,
            "message.part.updated" | "message.part.removed"
        )
    {
        for key in ["event_id", "id", "part_id"] {
            if let Some(value) = payload.get(key).and_then(Value::as_str) {
                if !value.is_empty() {
                    return Some(value.to_string());
                }
            }
        }
        for pointer in [
            "/payload/event_id",
            "/payload/id",
            "/payload/part_id",
            "/part/id",
            "/payload/part/id",
        ] {
            if let Some(value) = string_pointer(payload, pointer) {
                return Some(value);
            }
        }
        return None;
    }

    for key in ["event_id", "message_id", "turn_id", "id"] {
        if let Some(value) = payload.get(key).and_then(Value::as_str) {
            if !value.is_empty() {
                return Some(value.to_string());
            }
        }
    }
    for pointer in [
        "/payload/event_id",
        "/payload/message_id",
        "/payload/turn_id",
        "/payload/call_id",
        "/payload/id",
        "/payload/item/id",
        "/payload/item/message_id",
        "/payload/item/messageId",
        "/params/event_id",
        "/params/message_id",
        "/params/turn_id",
        "/params/call_id",
        "/params/id",
        "/params/item/id",
        "/params/item/message_id",
        "/params/item/messageId",
        "/item/id",
        "/item/message_id",
        "/item/messageId",
        "/message/id",
        "/messageID",
        "/uuid",
        "/attachment/toolUseID",
        "/toolUseID",
    ] {
        if let Some(value) = string_pointer(payload, pointer) {
            return Some(value);
        }
    }
    None
}

fn sequence_for_payload(
    tool: Tool,
    source_event_type: &str,
    payload: &Value,
    backfill_offset: Option<u64>,
) -> Option<i64> {
    for pointer in [
        "/sequence",
        "/index",
        "/ordinal",
        "/order",
        "/payload/sequence",
        "/payload/index",
        "/payload/ordinal",
        "/payload/order",
        "/params/sequence",
        "/params/index",
        "/params/ordinal",
        "/params/order",
    ] {
        if let Some(sequence) = i64_pointer(payload, pointer) {
            return Some(sequence);
        }
    }

    if tool == Tool::Codex {
        for pointer in [
            "/item_index",
            "/item_ordinal",
            "/turn_index",
            "/turn_ordinal",
            "/response_index",
            "/output_index",
            "/payload/item_index",
            "/payload/item_ordinal",
            "/payload/turn_index",
            "/payload/turn_ordinal",
            "/payload/response_index",
            "/payload/output_index",
            "/payload/item/index",
            "/payload/item/ordinal",
            "/payload/turn/index",
            "/payload/turn/ordinal",
            "/params/item_index",
            "/params/item_ordinal",
            "/params/turn_index",
            "/params/turn_ordinal",
            "/params/response_index",
            "/params/output_index",
            "/params/item/index",
            "/params/item/ordinal",
            "/params/turn/index",
            "/params/turn/ordinal",
            "/item/index",
            "/item/ordinal",
            "/turn/index",
            "/turn/ordinal",
        ] {
            if let Some(sequence) = i64_pointer(payload, pointer) {
                return Some(sequence);
            }
        }
    }

    if tool == Tool::Opencode
        && matches!(
            source_event_type,
            "message.part.updated" | "message.part.removed"
        )
    {
        for pointer in [
            "/part_index",
            "/part_sequence",
            "/payload/part_index",
            "/payload/part_sequence",
            "/part/index",
            "/part/sequence",
            "/payload/part/index",
            "/payload/part/sequence",
            "/params/part_index",
            "/params/part_sequence",
            "/params/part/index",
            "/params/part/sequence",
        ] {
            if let Some(sequence) = i64_pointer(payload, pointer) {
                return Some(sequence);
            }
        }
    }

    backfill_offset.and_then(|offset| i64::try_from(offset).ok())
}

fn i64_pointer(payload: &Value, pointer: &str) -> Option<i64> {
    let value = payload.pointer(pointer)?;
    value
        .as_i64()
        .or_else(|| value.as_u64().and_then(|value| i64::try_from(value).ok()))
        .or_else(|| {
            value
                .as_str()
                .and_then(|value| value.trim().parse::<i64>().ok())
        })
}

fn lock_path_for_raw_file(raw_file: &Path) -> PathBuf {
    raw_file.with_extension("jsonl.lock")
}

fn resolved_payload_for_envelope(raw_file: &Path, envelope: &EventEnvelope) -> Result<Value> {
    let Some(payload_ref) = envelope.payload_ref.as_deref() else {
        return Ok(envelope.payload.clone());
    };
    let Some(hash) = payload_ref.strip_prefix("sha256:") else {
        return Err(Error::Validation(format!(
            "unsupported payload_ref: {payload_ref}"
        )));
    };
    let blob_path = harness_home_for_raw_file(raw_file)
        .join("blobs")
        .join("sha256")
        .join(format!("{hash}.json"));
    let content = fs::read_to_string(&blob_path).map_err(|source| Error::Io {
        path: blob_path,
        source,
    })?;
    Ok(serde_json::from_str(&content)?)
}

fn harness_home_for_raw_file(raw_file: &Path) -> PathBuf {
    raw_file
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct SearchDocument {
    user_text: String,
    assistant_text: String,
    tool_intent: String,
    tool_output: String,
    metadata_text: String,
}

impl SearchDocument {
    fn render(&self) -> String {
        join_non_empty([
            self.user_text.as_str(),
            self.assistant_text.as_str(),
            self.tool_intent.as_str(),
            self.tool_output.as_str(),
            self.metadata_text.as_str(),
        ])
    }

    fn identity_text(&self) -> String {
        self.render()
    }
}

#[allow(dead_code)]
fn embedding_units_for_document(document: &SearchDocument) -> Vec<EmbeddingUnit> {
    let candidates = [
        (EmbeddingUnitKind::UserText, document.user_text.as_str()),
        (
            EmbeddingUnitKind::AssistantText,
            document.assistant_text.as_str(),
        ),
        (EmbeddingUnitKind::ToolIntent, document.tool_intent.as_str()),
        (
            EmbeddingUnitKind::MetadataText,
            document.metadata_text.as_str(),
        ),
    ];
    let mut units = Vec::new();
    for (kind, text) in candidates {
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        units.push(EmbeddingUnit {
            kind,
            unit_index: 0,
            text: text.to_string(),
            text_hash: sha256_hex(text.as_bytes()),
        });
    }
    units
}

fn search_document_for_event(canonical_type: CanonicalType, payload: &Value) -> SearchDocument {
    let mut document = SearchDocument::default();
    match canonical_type {
        CanonicalType::UserMessage => {
            document.user_text =
                preferred_text(payload, &["prompt", "text", "message", "content", "input"]);
        }
        CanonicalType::AssistantDelta | CanonicalType::AssistantMessage => {
            document.assistant_text =
                preferred_text(payload, &["text", "message", "content", "delta", "summary"]);
        }
        CanonicalType::ToolCall => {
            document.tool_intent = preferred_text(
                payload,
                &["tool_name", "command", "description", "input", "arguments"],
            );
        }
        CanonicalType::ToolResult => {
            document.tool_intent =
                preferred_text(payload, &["tool_name", "command", "status", "exit_code"]);
            document.tool_output =
                preferred_text(payload, &["output", "stderr", "stdout", "error", "result"]);
        }
        CanonicalType::FileChanged => {
            document.metadata_text =
                preferred_text(payload, &["file", "file_path", "path", "diff", "operation"]);
        }
        CanonicalType::CompactionBefore | CanonicalType::CompactionAfter => {
            document.assistant_text =
                preferred_text(payload, &["summary", "text", "content", "reason"]);
            document.metadata_text = preferred_text(payload, &["trigger"]);
        }
        CanonicalType::SessionEnded => {
            let text = preferred_text(payload, &["message", "reason", "summary", "usage_summary"]);
            let usage = scalar_text_for_keys(payload, &["usage", "usage_metadata"]);
            document.metadata_text = join_non_empty([text.as_str(), usage.as_str()]);
        }
        CanonicalType::Error => {
            document.metadata_text =
                preferred_text(payload, &["error", "message", "reason", "text", "details"]);
        }
        _ => {
            document.metadata_text = nonvolatile_text(payload);
        }
    }

    if document.render().trim().is_empty() {
        document.metadata_text = nonvolatile_text(payload);
    }
    document
}

fn message_text_for_document(canonical_type: CanonicalType, document: &SearchDocument) -> &str {
    match canonical_type {
        CanonicalType::UserMessage => &document.user_text,
        CanonicalType::AssistantDelta | CanonicalType::AssistantMessage => &document.assistant_text,
        CanonicalType::ToolResult => {
            if document.tool_output.trim().is_empty() {
                &document.tool_intent
            } else {
                &document.tool_output
            }
        }
        _ => "",
    }
}

fn identity_content_hash(canonical_type: CanonicalType, payload: &Value) -> Result<String> {
    let document = search_document_for_event(canonical_type, payload);
    let identity_text = normalize_identity_text(&document.identity_text());
    let bytes = if identity_text.is_empty() {
        serde_json::to_vec(&identity_payload(payload))?
    } else {
        identity_text.into_bytes()
    };
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    Ok(hex::encode(hasher.finalize()))
}

fn normalize_identity_text(text: &str) -> String {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn preferred_text(payload: &Value, keys: &[&str]) -> String {
    let mut values = Vec::new();
    collect_strings_for_keys(payload, keys, &mut values);
    join_owned(values)
}

fn scalar_text_for_keys(payload: &Value, keys: &[&str]) -> String {
    let mut values = Vec::new();
    collect_scalars_for_keys(payload, keys, &mut values);
    join_owned(values)
}

fn nonvolatile_text(payload: &Value) -> String {
    let mut values = Vec::new();
    collect_nonvolatile_strings(payload, &mut values);
    join_owned(values)
}

fn collect_strings_for_keys(value: &Value, keys: &[&str], output: &mut Vec<String>) {
    match value {
        Value::Object(map) => {
            for (key, value) in map {
                if keys.contains(&key.as_str()) {
                    collect_strings(value, output);
                } else {
                    collect_strings_for_keys(value, keys, output);
                }
            }
        }
        Value::Array(values) => {
            for value in values {
                collect_strings_for_keys(value, keys, output);
            }
        }
        _ => {}
    }
}

fn collect_scalars_for_keys(value: &Value, keys: &[&str], output: &mut Vec<String>) {
    match value {
        Value::Object(map) => {
            for (key, value) in map {
                if keys.contains(&key.as_str()) {
                    collect_scalar_key_values(value, key, output);
                } else {
                    collect_scalars_for_keys(value, keys, output);
                }
            }
        }
        Value::Array(values) => {
            for value in values {
                collect_scalars_for_keys(value, keys, output);
            }
        }
        _ => {}
    }
}

fn collect_scalar_key_values(value: &Value, key_hint: &str, output: &mut Vec<String>) {
    match value {
        Value::String(text) => push_clean(output, &format!("{key_hint} {text}")),
        Value::Number(number) => push_clean(output, &format!("{key_hint} {number}")),
        Value::Bool(flag) => push_clean(output, &format!("{key_hint} {flag}")),
        Value::Array(values) => {
            for value in values {
                collect_scalar_key_values(value, key_hint, output);
            }
        }
        Value::Object(map) => {
            for (key, value) in map {
                collect_scalar_key_values(value, key, output);
            }
        }
        Value::Null => {}
    }
}

fn collect_nonvolatile_strings(value: &Value, output: &mut Vec<String>) {
    match value {
        Value::String(text) => push_clean(output, text),
        Value::Array(values) => {
            for value in values {
                collect_nonvolatile_strings(value, output);
            }
        }
        Value::Object(map) => {
            for (key, value) in map {
                if !is_volatile_identity_key(key) {
                    collect_nonvolatile_strings(value, output);
                }
            }
        }
        _ => {}
    }
}

fn collect_strings(value: &Value, output: &mut Vec<String>) {
    match value {
        Value::String(text) => push_clean(output, text),
        Value::Array(values) => {
            for value in values {
                collect_strings(value, output);
            }
        }
        Value::Object(map) => {
            for (key, value) in map {
                if key == "type" {
                    continue;
                }
                collect_strings(value, output);
            }
        }
        _ => {}
    }
}

fn push_clean(output: &mut Vec<String>, text: &str) {
    let text = text.trim();
    if !text.is_empty() {
        output.push(text.to_string());
    }
}

fn join_owned(values: Vec<String>) -> String {
    let mut seen = BTreeSet::new();
    let mut output = Vec::new();
    for value in values {
        if seen.insert(value.clone()) {
            output.push(value);
        }
    }
    output.join("\n")
}

fn join_non_empty<'a>(values: impl IntoIterator<Item = &'a str>) -> String {
    values
        .into_iter()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn identity_payload(value: &Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.iter()
                .filter(|(key, _)| !is_volatile_identity_key(key))
                .map(|(key, value)| (key.clone(), identity_payload(value)))
                .collect(),
        ),
        Value::Array(values) => Value::Array(values.iter().map(identity_payload).collect()),
        _ => value.clone(),
    }
}

fn is_volatile_identity_key(key: &str) -> bool {
    matches!(
        key,
        "captured_at"
            | "timestamp"
            | "created_at"
            | "updated_at"
            | "tool_version"
            | "session_id"
            | "filename_session_id"
            | "project_root"
            | "cwd"
            | "source"
            | "source_event_type"
            | "source_event_id"
            | "hook_event_name"
            | "event"
            | "type"
            | "dedupe_key"
            | "raw_file"
            | "raw_offset"
            | "raw_line"
            | "message_id"
            | "event_id"
            | "turn_id"
            | "id"
    )
}

fn role_for(canonical_type: CanonicalType) -> Option<&'static str> {
    match canonical_type {
        CanonicalType::UserMessage => Some("user"),
        CanonicalType::AssistantDelta | CanonicalType::AssistantMessage => Some("assistant"),
        CanonicalType::ToolResult => Some("tool"),
        _ => None,
    }
}

fn tool_status_for(canonical_type: CanonicalType) -> Option<&'static str> {
    match canonical_type {
        CanonicalType::ToolCall => Some("started"),
        CanonicalType::ToolResult => Some("completed"),
        _ => None,
    }
}

fn compaction_state_for(canonical_type: CanonicalType) -> &'static str {
    match canonical_type {
        CanonicalType::CompactionBefore => "pre_compaction",
        CanonicalType::CompactionAfter => "post_compaction",
        _ => "none",
    }
}

fn string_field<'a>(payload: &'a Value, key: &str) -> Option<&'a str> {
    payload.get(key).and_then(Value::as_str)
}

fn set_if_exists(path: &Path, mode: u32) -> Result<()> {
    if path.exists() {
        chmod(path, mode)?;
    }
    Ok(())
}

#[cfg(unix)]
fn chmod(path: &Path, mode: u32) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let mut permissions = fs::metadata(path)
        .map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?
        .permissions();
    permissions.set_mode(mode);
    fs::set_permissions(path, permissions).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(not(unix))]
fn chmod(_path: &Path, _mode: u32) -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;
    use tempfile::tempdir;

    const SEMANTIC_RETRIEVAL_FIXTURE_JSON: &str =
        include_str!("../../../fixtures/semantic/retrieval.json");

    #[derive(Debug, Deserialize)]
    struct SemanticRetrievalFixture {
        schema_version: u32,
        tool: Tool,
        session_id: String,
        cwd: String,
        project_root: String,
        events: Vec<SemanticRetrievalEvent>,
        queries: Vec<SemanticRetrievalQuery>,
    }

    #[derive(Debug, Deserialize)]
    struct SemanticRetrievalEvent {
        event_id: String,
        role: String,
        text: String,
    }

    #[derive(Debug, Deserialize)]
    struct SemanticRetrievalQuery {
        query: String,
        relevant_event_ids: Vec<String>,
    }

    #[cfg(feature = "semantic")]
    struct FakeEmbedder {
        batch_size: usize,
        intra_threads: usize,
        fail_on_call: Option<usize>,
        calls: std::cell::Cell<usize>,
    }

    #[cfg(feature = "semantic")]
    impl FakeEmbedder {
        fn new(batch_size: usize, intra_threads: usize, fail_on_call: Option<usize>) -> Self {
            Self {
                batch_size,
                intra_threads,
                fail_on_call,
                calls: std::cell::Cell::new(0),
            }
        }
    }

    #[cfg(feature = "semantic")]
    impl Embedder for FakeEmbedder {
        fn embed_documents(&self, documents: &[String]) -> Result<Vec<Vec<f32>>> {
            let call = self.calls.get().saturating_add(1);
            self.calls.set(call);
            if self.fail_on_call == Some(call) {
                return Err(Error::SemanticUnavailable(format!(
                    "fake embed failure on call {call}"
                )));
            }
            Ok(documents
                .iter()
                .map(|document| {
                    let mut vector = vec![0.0; SEMANTIC_VECTOR_DIMENSIONS];
                    vector[0] = document.len().max(1) as f32;
                    vector
                })
                .collect())
        }

        fn embed_query(&self, _query: &str) -> Result<Vec<f32>> {
            let mut vector = vec![0.0; SEMANTIC_VECTOR_DIMENSIONS];
            vector[0] = 1.0;
            Ok(vector)
        }

        fn document_batch_size(&self) -> usize {
            self.batch_size
        }

        fn intra_threads(&self) -> usize {
            self.intra_threads
        }
    }

    #[test]
    fn session_id_sanitization_is_stable_and_filesystem_safe() {
        let unsafe_id = "thread/../with spaces:and:unicode-ç";
        let first = sanitize_session_id(unsafe_id);
        let second = sanitize_session_id(unsafe_id);

        assert_eq!(first, second);
        assert_eq!(first, "thread_.._with_spaces_and_unicode-_");
        assert!(first
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-')));
    }

    #[test]
    fn canonical_raw_path_uses_tool_and_sanitized_session_id() {
        let path = canonical_raw_path(Path::new("/tmp/harness"), Tool::Claude, "a/b c");

        assert_eq!(
            path,
            PathBuf::from("/tmp/harness/raw/claude/claude_a_b_c.jsonl")
        );
    }

    #[test]
    fn dedupe_key_generation_is_stable_across_source_and_time_metadata() {
        let payload_a = json!({
            "hook_event_name": "UserPromptSubmit",
            "captured_at": "2026-06-17T12:00:59Z",
            "prompt": "identity ignores observation metadata"
        });
        let payload_b = json!({
            "hook_event_name": "UserPromptSubmit",
            "captured_at": "2026-06-17T12:01:01Z",
            "prompt": "identity ignores observation metadata"
        });
        let first = dedupe_key(DedupeParts {
            tool: Tool::Codex,
            session_id: "session",
            canonical_type: CanonicalType::UserMessage,
            source_event_id: None,
            sequence: None,
            payload: &payload_a,
        })
        .unwrap();
        let second = dedupe_key(DedupeParts {
            tool: Tool::Codex,
            session_id: "session",
            canonical_type: CanonicalType::UserMessage,
            source_event_id: None,
            sequence: None,
            payload: &payload_b,
        })
        .unwrap();
        let third = dedupe_key(DedupeParts {
            tool: Tool::Codex,
            session_id: "session",
            canonical_type: CanonicalType::ToolResult,
            source_event_id: None,
            sequence: None,
            payload: &payload_a,
        })
        .unwrap();

        assert_eq!(first, second);
        assert_ne!(first, third);
        assert!(first.starts_with("sha256:"));
    }

    #[test]
    fn envelope_validation_rejects_invalid_enum_values() {
        for (field, value) in [
            ("tool", "bad-tool"),
            ("source", "bad-source"),
            ("canonical_type", "bad.type"),
        ] {
            let mut envelope = valid_envelope_json();
            envelope[field] = json!(value);

            let result = serde_json::from_value::<EventEnvelope>(envelope);
            assert!(result.is_err(), "{field} should reject {value}");
        }
    }

    #[test]
    fn envelope_validation_rejects_mismatched_filename_session_id() {
        let mut envelope: EventEnvelope = serde_json::from_value(valid_envelope_json()).unwrap();
        envelope.filename_session_id = "wrong".to_string();

        assert!(envelope.validate().is_err());
    }

    fn purge_opts(keep_model: bool, keep_config: bool, dry_run: bool) -> PurgeAllOptions {
        PurgeAllOptions {
            keep_model,
            keep_config,
            dry_run,
        }
    }

    #[test]
    fn purge_all_dry_run_removes_nothing() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();
        fs::write(home.join("raw/claude/x.jsonl"), b"{}\n").unwrap();

        let report = purge_all(&home, purge_opts(false, false, true)).unwrap();

        assert!(report.dry_run);
        assert!(home.join("raw").is_dir(), "dry run must not delete raw");
        assert!(home.join("index").is_dir());
        assert!(home.join("config.toml").is_file());
        assert!(report.authoritative_in_scope);
        assert_eq!(report.bytes_reclaimed, 0);
        assert!(report.bytes_in_scope > 0);
        let raw = report.artifacts.iter().find(|a| a.name == "raw").unwrap();
        assert_eq!(raw.action, PurgeAction::WouldRemove);
        assert_eq!(raw.tier, PurgeTier::Authoritative);
    }

    #[test]
    fn purge_all_removes_known_artifacts_but_keeps_home_and_foreign_files() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();
        fs::write(home.join("NOTES.txt"), b"keep me").unwrap();

        let report = purge_all(&home, purge_opts(false, false, false)).unwrap();

        assert!(home.is_dir(), "home directory itself must remain");
        for gone in PURGE_KNOWN_ENTRIES {
            assert!(!home.join(gone).exists(), "{gone} should be removed");
        }
        assert!(
            home.join("NOTES.txt").is_file(),
            "foreign files must be left untouched"
        );
        assert!(report
            .unknown_entries
            .iter()
            .any(|p| p.ends_with("NOTES.txt")));
        assert!(report.authoritative_in_scope);
        assert_eq!(
            report
                .artifacts
                .iter()
                .find(|a| a.name == "raw")
                .unwrap()
                .action,
            PurgeAction::Removed
        );
        assert!(report.bytes_reclaimed > 0);
    }

    #[test]
    fn purge_all_keep_model_and_config_preserves_them() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();
        fs::write(home.join("models/model.bin"), b"weights").unwrap();

        let report = purge_all(&home, purge_opts(true, true, false)).unwrap();

        assert!(home.join("models").is_dir(), "models kept");
        assert!(home.join("models/model.bin").is_file());
        assert!(home.join("config.toml").is_file(), "config kept");
        assert!(!home.join("raw").exists());
        assert!(!home.join("index").exists());
        assert_eq!(
            report
                .artifacts
                .iter()
                .find(|a| a.name == "models")
                .unwrap()
                .action,
            PurgeAction::Preserved
        );
        assert_eq!(
            report
                .artifacts
                .iter()
                .find(|a| a.name == "config.toml")
                .unwrap()
                .action,
            PurgeAction::Preserved
        );
    }

    #[test]
    fn purge_all_refuses_non_store_directory() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("not-a-store");
        fs::create_dir_all(&home).unwrap();
        fs::write(home.join("random.txt"), b"x").unwrap();

        let err = purge_all(&home, purge_opts(false, false, true)).unwrap_err();
        assert!(matches!(err, Error::Validation(_)));
        assert!(
            home.join("random.txt").is_file(),
            "nothing removed on refusal"
        );
    }

    #[test]
    fn purge_all_missing_home_is_idempotent_noop() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("never-created");

        let report = purge_all(&home, purge_opts(false, false, false)).unwrap();
        assert!(report.artifacts.is_empty());
        assert_eq!(report.bytes_reclaimed, 0);
        assert!(!report.authoritative_in_scope);
    }

    #[cfg(unix)]
    #[test]
    fn purge_all_removes_model_symlink_not_its_target() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();
        let external = temp.path().join("external-models");
        fs::create_dir_all(&external).unwrap();
        fs::write(external.join("weights.bin"), b"important").unwrap();
        fs::remove_dir_all(home.join("models")).unwrap();
        std::os::unix::fs::symlink(&external, home.join("models")).unwrap();

        let report = purge_all(&home, purge_opts(false, false, false)).unwrap();

        assert!(!home.join("models").exists(), "symlink unlinked");
        assert!(external.is_dir(), "symlink target preserved");
        assert!(
            external.join("weights.bin").is_file(),
            "target contents preserved"
        );
        assert_eq!(
            report
                .artifacts
                .iter()
                .find(|a| a.name == "models")
                .unwrap()
                .action,
            PurgeAction::Removed
        );
    }

    #[test]
    fn init_home_creates_required_layout_and_valid_sqlite_database() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");

        let report = init_home(&home).unwrap();

        for relative in [
            ".",
            "raw",
            "raw/codex",
            "raw/claude",
            "raw/opencode",
            "spool",
            "spool/dedupe",
            "checkpoints",
            "blobs/sha256",
            "models",
            "logs",
            "backups",
        ] {
            assert!(home.join(relative).is_dir(), "{relative} should exist");
            assert_private_dir_mode(&home.join(relative));
        }
        assert_private_file_mode(&home.join("config.toml"));
        assert!(report.db_path.is_file());
        assert_private_file_mode(&report.db_path);
        if report.db_path.with_file_name("harness.db-wal").exists() {
            assert_private_file_mode(&report.db_path.with_file_name("harness.db-wal"));
        }
        if report.db_path.with_file_name("harness.db-shm").exists() {
            assert_private_file_mode(&report.db_path.with_file_name("harness.db-shm"));
        }

        let conn = Connection::open(&report.db_path).unwrap();
        let integrity: String = conn
            .query_row("PRAGMA integrity_check;", [], |row| row.get(0))
            .unwrap();
        let user_version: i64 = conn
            .query_row("PRAGMA user_version;", [], |row| row.get(0))
            .unwrap();

        assert_eq!(integrity, "ok");
        assert_eq!(user_version, 1);
        assert_eq!(opencode_server_url(&home).unwrap(), None);
    }

    #[cfg(feature = "semantic")]
    #[test]
    fn semantic_feature_loads_sqlite_vec_with_bundled_rusqlite() {
        unsafe {
            rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
                sqlite_vec::sqlite3_vec_init as *const (),
            )));
        }

        let conn = Connection::open_in_memory().unwrap();
        let version: String = conn
            .query_row("select vec_version()", [], |row| row.get(0))
            .unwrap();
        assert!(version.starts_with('v'), "{version}");

        conn.execute(
            "create virtual table vectors using vec0(embedding float[4])",
            [],
        )
        .unwrap();
    }

    #[cfg(feature = "semantic")]
    #[test]
    fn semantic_feature_initializes_derived_vector_schema() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();

        let db_path = home.join("index").join("harness.db");
        let conn = open_index(&db_path).unwrap();

        assert!(table_exists(&conn, &db_path, "vector_units").unwrap());
        assert!(table_exists(&conn, &db_path, "vector_unit_texts").unwrap());
        assert!(table_exists(&conn, &db_path, "vector_unit_embeddings").unwrap());
        assert_eq!(
            table_count(&conn, &db_path, "vector_unit_embeddings").unwrap(),
            0
        );
        let version: String = conn
            .query_row("select vec_version()", [], |row| row.get(0))
            .unwrap();
        assert!(version.starts_with('v'), "{version}");

        let footprint = storage_footprint(&home);
        assert_eq!(footprint.vectors_bytes, 0);
        assert!(footprint.index_bytes > 0);
    }

    #[test]
    fn embeddinggemma_prompt_prefixes_are_pinned() {
        assert_eq!(
            query_embedding_input("  auth bug  "),
            "task: search result | query: auth bug"
        );
        assert_eq!(
            document_embedding_input("  fixed login timeout  "),
            "title: none | text: fixed login timeout"
        );
        assert_ne!(
            query_embedding_input("same text"),
            document_embedding_input("same text")
        );
    }

    #[cfg(feature = "semantic")]
    #[test]
    fn semantic_vectors_persist_in_sqlite_vec() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();

        let db_path = home.join("index").join("harness.db");
        let conn = open_index(&db_path).unwrap();
        ensure_semantic_vector_schema(&conn, &db_path).unwrap();

        let mut vector = vec![0.0_f32; SEMANTIC_VECTOR_DIMENSIONS];
        vector[0] = 1.0;
        let vector_blob = vector_to_blob(&vector).unwrap();
        conn.execute(
            "INSERT INTO vector_unit_embeddings(unit_id, embedding) VALUES (?1, ?2)",
            params![1_i64, vector_blob.clone()],
        )
        .unwrap();

        let unit_id: i64 = conn
            .query_row(
                "SELECT unit_id FROM vector_unit_embeddings
                 WHERE embedding MATCH ?1 AND k = 1
                 ORDER BY distance",
                params![vector_blob],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(unit_id, 1);
        assert_eq!(storage_footprint(&home).vectors_bytes, 1024);
    }

    #[cfg(feature = "semantic")]
    #[test]
    fn semantic_index_materializes_units_without_model_or_payload_duplication() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();
        ingest_hook_event(
            &home,
            Tool::Claude,
            json!({
                "session_id": "semantic-units",
                "hook_event_name": "UserPromptSubmit",
                "message_id": "semantic-units-1",
                "prompt": "remember the fuzzy auth regression fix",
                "cwd": "/tmp/nabu-fixture",
                "project_root": "/tmp/nabu-fixture"
            }),
        )
        .unwrap();

        index_once(&home).unwrap();
        let db_path = home.join("index").join("harness.db");
        let conn = open_index(&db_path).unwrap();

        assert_eq!(table_count(&conn, &db_path, "vector_units").unwrap(), 1);
        assert_eq!(
            table_count(&conn, &db_path, "vector_unit_texts").unwrap(),
            1
        );
        assert_eq!(
            table_count(&conn, &db_path, "vector_unit_embeddings").unwrap(),
            0
        );
        let payload_json: Option<String> = conn
            .query_row("SELECT payload_json FROM events LIMIT 1", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert!(payload_json.is_none());
        assert!(!embedding_model_status(&home).semantic_available);
    }

    #[test]
    fn opencode_hook_resolves_session_id_from_native_event_shapes() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();

        // Message/part/tool/etc. events carry top-level `sessionID`; live plugin
        // payloads may nest the object under `info`/`part`.
        let by_session_id = ingest_hook_event(
            &home,
            Tool::Opencode,
            json!({
                "hook_event_name": "message.updated",
                "id": "msg_abc",
                "sessionID": "ses_top_level",
                "role": "assistant"
            }),
        )
        .unwrap();
        assert!(by_session_id.appended);
        assert!(by_session_id
            .raw_file
            .to_string_lossy()
            .contains("ses_top_level"));

        let nested_part = ingest_hook_event(
            &home,
            Tool::Opencode,
            json!({
                "hook_event_name": "message.part.updated",
                "part": { "id": "prt_1", "sessionID": "ses_nested_part", "type": "text" }
            }),
        )
        .unwrap();
        assert!(nested_part
            .raw_file
            .to_string_lossy()
            .contains("ses_nested_part"));

        // `session.*` events have no `sessionID`; the session id is `id`.
        let session_created = ingest_hook_event(
            &home,
            Tool::Opencode,
            json!({
                "hook_event_name": "session.created",
                "id": "ses_from_id",
                "directory": "/tmp/project"
            }),
        )
        .unwrap();
        assert!(session_created
            .raw_file
            .to_string_lossy()
            .contains("ses_from_id"));
    }

    #[test]
    fn opencode_hook_does_not_mistake_message_id_for_session_id() {
        // A non-session event with `id` but no `sessionID` must NOT fall back to
        // `id` (that would be the message id, not the session id).
        let payload = json!({
            "hook_event_name": "message.updated",
            "id": "msg_no_session"
        });
        let result = opencode_hook_session_id(&payload, "message.updated");
        assert!(matches!(result, Err(Error::Validation(_))));
    }

    #[test]
    fn opencode_hook_rejects_event_without_resolvable_session_id() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();
        let result = ingest_hook_event(
            &home,
            Tool::Opencode,
            json!({ "hook_event_name": "file.edited", "filename": "src/lib.rs" }),
        );
        assert!(matches!(result, Err(Error::Validation(_))));
    }

    #[cfg(feature = "semantic")]
    #[test]
    fn semantic_index_no_embed_skips_fake_model_and_leaves_vectors_empty() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();
        write_fake_semantic_model_files(&home);
        ingest_hook_event(
            &home,
            Tool::Claude,
            json!({
                "session_id": "semantic-no-embed",
                "hook_event_name": "UserPromptSubmit",
                "message_id": "semantic-no-embed-1",
                "prompt": "deferred semantic fake model marker",
                "cwd": "/tmp/nabu-fixture",
                "project_root": "/tmp/nabu-fixture"
            }),
        )
        .unwrap();

        let report = index_once_with_options(&home, IndexOptions { embed: false }).unwrap();
        let db_path = home.join("index").join("harness.db");
        let conn = open_index(&db_path).unwrap();

        assert_eq!(report.indexed_events, 1);
        assert_eq!(
            search_history(&home, "deferred semantic fake model", 10)
                .unwrap()
                .len(),
            1
        );
        assert_eq!(table_count(&conn, &db_path, "vector_units").unwrap(), 1);
        assert_eq!(
            table_count(&conn, &db_path, "vector_unit_embeddings").unwrap(),
            0
        );
        assert!(!embedding_model_status(&home).semantic_available);
    }

    #[cfg(feature = "semantic")]
    #[test]
    fn semantic_index_discloses_unembedded_count_before_model_load() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();
        write_fake_semantic_model_files(&home);
        ingest_hook_event(
            &home,
            Tool::Claude,
            json!({
                "session_id": "semantic-plan",
                "hook_event_name": "UserPromptSubmit",
                "message_id": "semantic-plan-1",
                "prompt": "semantic plan progress marker",
                "cwd": "/tmp/nabu-fixture",
                "project_root": "/tmp/nabu-fixture"
            }),
        )
        .unwrap();

        let mut progress = Vec::new();
        let result =
            index_once_with_options_and_progress(&home, IndexOptions::default(), |event| {
                progress.push(event)
            });

        assert!(
            result.is_err(),
            "fake model files should make model loading fail after the plan is emitted"
        );
        assert_eq!(progress.first().unwrap().phase, "embedding_plan");
        assert_eq!(progress.first().unwrap().status, "ready");
        assert_eq!(progress.first().unwrap().total_units, 1);
        assert!(progress
            .iter()
            .any(|event| event.phase == "loading_model" && event.status == "started"));
    }

    #[cfg(feature = "semantic")]
    #[test]
    fn semantic_collect_uses_compact_unit_texts_and_backfills_legacy_rows() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();
        ingest_hook_event(
            &home,
            Tool::Claude,
            json!({
                "session_id": "semantic-texts",
                "hook_event_name": "UserPromptSubmit",
                "message_id": "semantic-texts-1",
                "prompt": "compact vector unit text marker",
                "cwd": "/tmp/nabu-fixture",
                "project_root": "/tmp/nabu-fixture"
            }),
        )
        .unwrap();
        index_once(&home).unwrap();

        let db_path = home.join("index").join("harness.db");
        let conn = open_index(&db_path).unwrap();
        assert_eq!(
            table_count(&conn, &db_path, "vector_unit_texts").unwrap(),
            1
        );
        conn.execute("DELETE FROM vector_unit_texts", []).unwrap();
        assert_eq!(
            table_count(&conn, &db_path, "vector_unit_texts").unwrap(),
            0
        );

        let units = collect_unembedded_units(&conn, &db_path).unwrap();
        assert_eq!(units.len(), 1);
        assert!(units[0].text.contains("compact vector unit text marker"));
        assert_eq!(
            table_count(&conn, &db_path, "vector_unit_texts").unwrap(),
            1
        );
    }

    #[cfg(feature = "semantic")]
    #[test]
    fn semantic_index_does_not_load_fake_model_when_no_units_need_embedding() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();
        write_fake_semantic_model_files(&home);

        let mut progress = Vec::new();
        let embedded =
            embed_index_if_available_with_progress(&home, |event| progress.push(event)).unwrap();

        assert_eq!(embedded, 0);
        assert!(progress.is_empty());
    }

    #[cfg(feature = "semantic")]
    #[test]
    fn semantic_embedding_batches_by_length_and_streams_progress() {
        let mut units = vec![
            UnembeddedUnit {
                unit_id: 2,
                text: "x ".repeat(200),
                estimated_tokens: 200,
            },
            UnembeddedUnit {
                unit_id: 3,
                text: "tiny".to_string(),
                estimated_tokens: estimated_embedding_token_count("tiny"),
            },
            UnembeddedUnit {
                unit_id: 1,
                text: "short unit".to_string(),
                estimated_tokens: estimated_embedding_token_count("short unit"),
            },
        ];
        bucket_unembedded_units(&mut units);
        assert_eq!(
            units.iter().map(|unit| unit.unit_id).collect::<Vec<_>>(),
            vec![1, 3, 2]
        );

        let progress = embedding_index_progress(
            "embedding",
            "running",
            50,
            100,
            Instant::now() - StdDuration::from_secs(2),
            &FakeEmbedder::new(64, 8, None),
            2048,
        );
        assert_eq!(progress.batch_size, 64);
        assert_eq!(progress.write_chunk_size, 2048);
        assert_eq!(progress.intra_threads, 8);
        assert!(progress.units_per_second > 0.0);
        assert!(progress.eta_seconds.is_some());
    }

    #[cfg(feature = "semantic")]
    #[test]
    fn semantic_embedding_writes_commit_in_resumable_chunks() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();
        for index in 0..3 {
            ingest_hook_event(
                &home,
                Tool::Claude,
                json!({
                    "session_id": "semantic-resume",
                    "hook_event_name": "UserPromptSubmit",
                    "message_id": format!("semantic-resume-{index}"),
                    "sequence": index,
                    "prompt": format!("semantic resumable embedding unit {index}"),
                    "cwd": "/tmp/nabu-fixture",
                    "project_root": "/tmp/nabu-fixture"
                }),
            )
            .unwrap();
        }
        index_once(&home).unwrap();

        let db_path = home.join("index").join("harness.db");
        let mut conn = open_index(&db_path).unwrap();
        assert_eq!(table_count(&conn, &db_path, "vector_units").unwrap(), 3);

        let failing = FakeEmbedder::new(1, 4, Some(3));
        let mut failed_progress = Vec::new();
        let result = embed_unembedded_units_with_config(
            &mut conn,
            &db_path,
            &failing,
            EmbeddingWriteConfig {
                write_chunk_size: 2,
            },
            |event| failed_progress.push(event),
        );
        assert!(result.is_err());
        assert_eq!(
            table_count(&conn, &db_path, "vector_unit_embeddings").unwrap(),
            2
        );
        assert_eq!(failed_progress.first().unwrap().status, "started");

        let succeeding = FakeEmbedder::new(1, 4, None);
        let mut resumed_progress = Vec::new();
        let embedded = embed_unembedded_units_with_config(
            &mut conn,
            &db_path,
            &succeeding,
            EmbeddingWriteConfig {
                write_chunk_size: 2,
            },
            |event| resumed_progress.push(event),
        )
        .unwrap();
        assert_eq!(embedded, 1);
        assert_eq!(
            table_count(&conn, &db_path, "vector_unit_embeddings").unwrap(),
            3
        );
        assert_eq!(resumed_progress.first().unwrap().status, "started");
        assert_eq!(resumed_progress.last().unwrap().status, "completed");
        assert_eq!(resumed_progress.last().unwrap().embedded_units, 1);
        assert_eq!(resumed_progress.last().unwrap().total_units, 1);
    }

    #[cfg(feature = "semantic")]
    #[test]
    fn semantic_no_embed_defers_vectors_until_later_default_index_when_model_present() {
        let Some(model_home) = semantic_test_model_home() else {
            eprintln!("skipping semantic no-embed acceptance: local model cache not present");
            return;
        };
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();
        attach_semantic_model_cache(&home, &model_home);
        ingest_hook_event(
            &home,
            Tool::Claude,
            json!({
                "session_id": "semantic-deferred-real-model",
                "hook_event_name": "UserPromptSubmit",
                "message_id": "semantic-deferred-real-model-1",
                "prompt": "defer semantic embedding until the later default index pass",
                "cwd": "/tmp/nabu-fixture",
                "project_root": "/tmp/nabu-fixture"
            }),
        )
        .unwrap();

        let first = index_once_with_options(&home, IndexOptions { embed: false }).unwrap();
        let db_path = home.join("index").join("harness.db");
        let conn = open_index(&db_path).unwrap();
        assert_eq!(first.indexed_events, 1);
        assert_eq!(table_count(&conn, &db_path, "vector_units").unwrap(), 1);
        assert_eq!(
            table_count(&conn, &db_path, "vector_unit_embeddings").unwrap(),
            0
        );
        assert!(!embedding_model_status(&home).semantic_available);

        let mut progress = Vec::new();
        let second =
            index_once_with_options_and_progress(&home, IndexOptions::default(), |event| {
                progress.push(event)
            })
            .unwrap();
        assert_eq!(second.indexed_events, 0);
        assert!(
            table_count(&conn, &db_path, "vector_unit_embeddings").unwrap() > 0,
            "default index should embed units deferred by --no-embed"
        );
        assert!(embedding_model_status(&home).semantic_available);
        assert_eq!(progress.first().unwrap().phase, "embedding_plan");
        assert_eq!(progress.first().unwrap().total_units, 1);
    }

    #[cfg(feature = "semantic")]
    #[test]
    fn hybrid_beats_lexical_on_labeled_retrieval_fixture_when_model_present() {
        let Some(model_home) = semantic_test_model_home() else {
            eprintln!(
                "skipping semantic retrieval quality acceptance: local model cache not present"
            );
            return;
        };
        let fixture = semantic_retrieval_fixture();
        assert!(!fixture.events.is_empty());
        assert!(!fixture.queries.is_empty());

        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();
        attach_semantic_model_cache(&home, &model_home);
        seed_semantic_retrieval_fixture(&home, &fixture);
        index_once(&home).unwrap();
        assert!(embedding_model_status(&home).semantic_available);

        let k = 3usize;
        let first_results = hybrid_result_ids_by_query(&home, &fixture, k);
        let first_vectors = vector_snapshot(&home);
        assert!(!first_vectors.is_empty());

        let mut strict_wins = 0usize;
        let mut aggregate_lexical_precision = 0.0;
        let mut aggregate_hybrid_precision = 0.0;
        let mut aggregate_lexical_recall = 0.0;
        let mut aggregate_hybrid_recall = 0.0;
        for query in &fixture.queries {
            let relevant = query
                .relevant_event_ids
                .iter()
                .cloned()
                .collect::<BTreeSet<_>>();
            let lexical = result_event_ids(
                &home,
                search_history_page(
                    &home,
                    &query.query,
                    SearchOptions {
                        mode: SearchMode::Lexical,
                        limit: k,
                        dedupe: false,
                        ..SearchOptions::default()
                    },
                )
                .unwrap()
                .results,
            );
            let hybrid = result_event_ids(
                &home,
                search_history_page(
                    &home,
                    &query.query,
                    SearchOptions {
                        mode: SearchMode::Hybrid,
                        limit: k,
                        dedupe: false,
                        ..SearchOptions::default()
                    },
                )
                .unwrap()
                .results,
            );
            let lexical_precision = precision_at_k(&lexical, &relevant, k);
            let hybrid_precision = precision_at_k(&hybrid, &relevant, k);
            let lexical_recall = recall_at_k(&lexical, &relevant, k);
            let hybrid_recall = recall_at_k(&hybrid, &relevant, k);
            aggregate_lexical_precision += lexical_precision;
            aggregate_hybrid_precision += hybrid_precision;
            aggregate_lexical_recall += lexical_recall;
            aggregate_hybrid_recall += hybrid_recall;

            eprintln!(
                "semantic fixture query={:?} lexical_ids={:?} hybrid_ids={:?} precision@{} lexical={:.3} hybrid={:.3} recall@{} lexical={:.3} hybrid={:.3}",
                query.query,
                lexical,
                hybrid,
                k,
                lexical_precision,
                hybrid_precision,
                k,
                lexical_recall,
                hybrid_recall
            );
            assert!(
                hybrid_precision >= lexical_precision,
                "hybrid precision regressed for query {:?}: lexical {:?} ({:.3}), hybrid {:?} ({:.3})",
                query.query,
                lexical,
                lexical_precision,
                hybrid,
                hybrid_precision
            );
            assert!(
                hybrid_recall >= lexical_recall,
                "hybrid recall regressed for query {:?}: lexical {:?} ({:.3}), hybrid {:?} ({:.3})",
                query.query,
                lexical,
                lexical_recall,
                hybrid,
                hybrid_recall
            );
            if hybrid_precision > lexical_precision || hybrid_recall > lexical_recall {
                strict_wins += 1;
            }
        }
        let query_count = fixture.queries.len() as f64;
        eprintln!(
            "semantic fixture aggregate precision@{} lexical={:.3} hybrid={:.3} recall@{} lexical={:.3} hybrid={:.3} strict_wins={}/{}",
            k,
            aggregate_lexical_precision / query_count,
            aggregate_hybrid_precision / query_count,
            k,
            aggregate_lexical_recall / query_count,
            aggregate_hybrid_recall / query_count,
            strict_wins,
            fixture.queries.len()
        );
        assert!(
            strict_wins > 0,
            "hybrid tied lexical on every labeled semantic query; this does not prove the M5 retrieval-quality win"
        );

        remove_index_database(&home);
        index_once(&home).unwrap();
        let second_vectors = vector_snapshot(&home);
        let second_results = hybrid_result_ids_by_query(&home, &fixture, k);

        assert_eq!(first_vectors, second_vectors);
        assert_eq!(first_results, second_results);
    }

    fn semantic_retrieval_fixture() -> SemanticRetrievalFixture {
        serde_json::from_str(SEMANTIC_RETRIEVAL_FIXTURE_JSON)
            .expect("semantic retrieval fixture must be valid JSON")
    }

    #[cfg(feature = "semantic")]
    fn write_fake_semantic_model_files(home: &Path) {
        let model_root = semantic_model_cache_path(home);
        for (_, local) in SEMANTIC_MODEL_REMOTE_FILES {
            let path = model_root.join(local);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, b"not a real model").unwrap();
        }
    }

    #[cfg(feature = "semantic")]
    fn semantic_test_model_home() -> Option<PathBuf> {
        if let Ok(model_dir) = std::env::var("NABU_SEMANTIC_MODEL_DIR") {
            let model_dir = PathBuf::from(model_dir);
            if semantic_model_files_present_at(&model_dir) {
                return Some(model_dir);
            }
        }

        let mut candidates = Vec::new();
        if let Ok(home) = std::env::var("NABU_SEMANTIC_TEST_HOME") {
            candidates.push(PathBuf::from(home));
        }
        if let Ok(home) = std::env::var("NABU_HOME") {
            candidates.push(PathBuf::from(home));
        }
        if let Ok(home) = resolve_home(None) {
            candidates.push(home);
        }

        candidates.into_iter().find_map(|home| {
            let cache_path = semantic_model_cache_path(&home);
            semantic_model_files_present_at(&cache_path).then_some(cache_path)
        })
    }

    #[cfg(feature = "semantic")]
    fn semantic_model_files_present_at(cache_path: &Path) -> bool {
        SEMANTIC_MODEL_REMOTE_FILES
            .iter()
            .all(|(_, local)| cache_path.join(local).is_file())
    }

    #[cfg(feature = "semantic")]
    fn attach_semantic_model_cache(home: &Path, source_cache_path: &Path) {
        let model_root = home.join("models");
        fs::create_dir_all(&model_root).unwrap();
        let target = semantic_model_cache_path(home);
        if target.exists() {
            return;
        }
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(source_cache_path, &target).unwrap();
        }
        #[cfg(not(unix))]
        {
            let _ = source_cache_path;
            panic!("semantic model cache symlink test requires a Unix platform");
        }
    }

    #[cfg(feature = "semantic")]
    fn seed_semantic_retrieval_fixture(home: &Path, fixture: &SemanticRetrievalFixture) {
        for (index, event) in fixture.events.iter().enumerate() {
            let mut payload = json!({
                "session_id": fixture.session_id,
                "message_id": event.event_id,
                "cwd": fixture.cwd,
                "project_root": fixture.project_root,
            });
            match event.role.as_str() {
                "user" => {
                    payload["hook_event_name"] = json!("UserPromptSubmit");
                    payload["prompt"] = json!(event.text);
                }
                "assistant" => {
                    payload["hook_event_name"] = json!("MessageDisplay");
                    payload["text"] = json!(event.text);
                    payload["index"] = json!(index as i64);
                    payload["final"] = json!(true);
                }
                role => panic!("unsupported semantic fixture role: {role}"),
            }
            ingest_hook_event(home, Tool::Claude, payload).unwrap();
        }
    }

    #[cfg(feature = "semantic")]
    fn result_event_ids(home: &Path, results: Vec<SearchResult>) -> Vec<String> {
        results
            .iter()
            .map(|result| result_event_id(home, result))
            .collect()
    }

    #[cfg(feature = "semantic")]
    fn result_event_id(home: &Path, result: &SearchResult) -> String {
        let db_path = home.join("index").join("harness.db");
        let conn = open_index(&db_path).unwrap();
        conn.query_row(
            "SELECT COALESCE(message_id, source_event_id, CAST(id AS TEXT))
             FROM events
             WHERE tool = ?1
               AND session_id = ?2
               AND raw_file = ?3
               AND raw_line = ?4
             ORDER BY id
             LIMIT 1",
            params![
                result.tool.as_str(),
                &result.session_id,
                &result.raw_file,
                result.raw_line,
            ],
            |row| row.get(0),
        )
        .unwrap()
    }

    #[cfg(feature = "semantic")]
    fn precision_at_k(ids: &[String], relevant: &BTreeSet<String>, k: usize) -> f64 {
        if k == 0 {
            return 0.0;
        }
        relevant_hits_at_k(ids, relevant, k) as f64 / k as f64
    }

    #[cfg(feature = "semantic")]
    fn recall_at_k(ids: &[String], relevant: &BTreeSet<String>, k: usize) -> f64 {
        if relevant.is_empty() {
            return 0.0;
        }
        relevant_hits_at_k(ids, relevant, k) as f64 / relevant.len() as f64
    }

    #[cfg(feature = "semantic")]
    fn relevant_hits_at_k(ids: &[String], relevant: &BTreeSet<String>, k: usize) -> usize {
        ids.iter()
            .take(k)
            .filter(|event_id| relevant.contains(*event_id))
            .count()
    }

    #[cfg(feature = "semantic")]
    fn hybrid_result_ids_by_query(
        home: &Path,
        fixture: &SemanticRetrievalFixture,
        k: usize,
    ) -> Vec<Vec<String>> {
        fixture
            .queries
            .iter()
            .map(|query| {
                result_event_ids(
                    home,
                    search_history_page(
                        home,
                        &query.query,
                        SearchOptions {
                            mode: SearchMode::Hybrid,
                            limit: k,
                            dedupe: false,
                            ..SearchOptions::default()
                        },
                    )
                    .unwrap()
                    .results,
                )
            })
            .collect()
    }

    #[cfg(feature = "semantic")]
    fn vector_snapshot(home: &Path) -> Vec<(i64, Vec<u8>)> {
        let db_path = home.join("index").join("harness.db");
        let conn = open_index(&db_path).unwrap();
        let mut statement = conn
            .prepare("SELECT unit_id, embedding FROM vector_unit_embeddings ORDER BY unit_id")
            .unwrap();
        let rows = statement
            .query_map([], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))
            })
            .unwrap();
        rows.map(|row| row.unwrap()).collect()
    }

    #[cfg(feature = "semantic")]
    fn remove_index_database(home: &Path) {
        let db_path = home.join("index").join("harness.db");
        for path in [
            db_path.clone(),
            db_path.with_file_name("harness.db-wal"),
            db_path.with_file_name("harness.db-shm"),
        ] {
            match fs::remove_file(&path) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => panic!("failed to remove {}: {error}", path.display()),
            }
        }
    }

    #[test]
    fn opencode_server_url_reads_config_toml_key() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();
        fs::write(
            home.join("config.toml"),
            "schema_version = 1\n\n[opencode]\nserver_url = \"http://127.0.0.1:4096\"\n",
        )
        .unwrap();

        assert_eq!(
            opencode_server_url(&home).unwrap(),
            Some("http://127.0.0.1:4096".to_string())
        );
    }

    #[cfg(unix)]
    fn assert_private_dir_mode(path: &Path) {
        use std::os::unix::fs::PermissionsExt;

        let mode = fs::metadata(path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o700, "{} should be 0700", path.display());
    }

    #[cfg(not(unix))]
    fn assert_private_dir_mode(_path: &Path) {}

    #[cfg(unix)]
    fn assert_private_file_mode(path: &Path) {
        use std::os::unix::fs::PermissionsExt;

        let mode = fs::metadata(path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "{} should be 0600", path.display());
    }

    #[cfg(not(unix))]
    fn assert_private_file_mode(_path: &Path) {}

    #[test]
    fn redaction_rules_match_contract_and_preserve_safe_text() {
        let fixture = include_str!("../../../fixtures/redaction/secrets.txt");
        let redacted = redact_export_text(fixture);

        for expected in [
            "[REDACTED:PRIVATE_KEY]",
            "Bearer [REDACTED:BEARER_TOKEN]",
            "[REDACTED:API_KEY]",
            "DATABASE_PASSWORD=[REDACTED:ENV_VALUE]",
        ] {
            assert!(redacted.contains(expected), "{expected}");
        }
        for secret in [
            "private-key-material",
            "abcdefghijklmnopqrstuvwxyz123456",
            "supersecretvalue",
            "AKIA1234567890ABCDEF",
        ] {
            assert!(!redacted.contains(secret), "{secret}");
        }
        assert!(redacted.contains("redaction fixture marker keeps safe surrounding text"));
        assert!(redacted.contains("redaction fixture marker keeps trailing safe text"));
    }

    #[test]
    fn oversized_payloads_are_spilled_and_indexed_from_blob() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();
        let prompt =
            "oversized payload fixture marker ".repeat((MAX_INLINE_ENVELOPE_BYTES / 32) + 1024);

        ingest_hook_event(
            &home,
            Tool::Claude,
            json!({
                "session_id": "fixture-session",
                "hook_event_name": "UserPromptSubmit",
                "message_id": "oversized-payload-1",
                "cwd": "/tmp/nabu-fixture",
                "project_root": "/tmp/nabu-fixture",
                "prompt": prompt
            }),
        )
        .unwrap();

        let raw_path = canonical_raw_path(&home, Tool::Claude, "fixture-session");
        let raw = fs::read_to_string(&raw_path).unwrap();
        let envelope: EventEnvelope = serde_json::from_str(raw.trim_end()).unwrap();
        let payload_ref = envelope.payload_ref.as_deref().unwrap();
        assert!(payload_ref.starts_with("sha256:"));
        assert!(envelope.payload.is_null());
        let hash = payload_ref.trim_start_matches("sha256:");
        assert!(home
            .join("blobs")
            .join("sha256")
            .join(format!("{hash}.json"))
            .is_file());
        assert!(raw.len() < MAX_INLINE_ENVELOPE_BYTES);

        index_once(&home).unwrap();
        let page = search_history_page(
            &home,
            "oversized payload fixture marker",
            SearchOptions {
                limit: 1,
                include_payload: true,
                dedupe: false,
                max_snippet_chars: 80,
                ..SearchOptions::default()
            },
        )
        .unwrap();
        assert_eq!(page.results.len(), 1);
        assert_eq!(page.results[0].session_id, "fixture-session");
        assert!(page.results[0]
            .payload
            .get("prompt")
            .and_then(Value::as_str)
            .unwrap()
            .contains("oversized payload fixture marker"));
    }

    #[test]
    fn markdown_export_includes_sensitivity_warning() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();
        ingest_hook_event(
            &home,
            Tool::Claude,
            json!({
                "session_id": "warning-session",
                "hook_event_name": "UserPromptSubmit",
                "message_id": "warning-1",
                "cwd": "/tmp/nabu-fixture",
                "project_root": "/tmp/nabu-fixture",
                "prompt": "markdown warning marker"
            }),
        )
        .unwrap();
        index_once(&home).unwrap();

        let full =
            export_session_markdown_with_options(&home, Tool::Claude, "warning-session", false)
                .unwrap();
        let redacted =
            export_session_markdown_with_options(&home, Tool::Claude, "warning-session", true)
                .unwrap();

        assert!(full.contains("Sensitivity: this export is not redacted."));
        assert!(redacted.contains("Sensitivity: redacted export."));
    }

    #[test]
    fn raw_append_and_index_dedupe_same_native_event_across_sources() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();
        let payload = json!({
            "session_id": "fixture-session",
            "hook_event_name": "UserPromptSubmit",
            "event_id": "same-native-event-1",
            "cwd": "/tmp/nabu-fixture",
            "project_root": "/tmp/nabu-fixture",
            "prompt": "cross source dedupe fixture marker"
        });

        let first = ingest_hook_event(&home, Tool::Codex, payload.clone()).unwrap();
        let backfill_event = envelope_from_backfill_payload(
            Tool::Codex,
            Path::new("/tmp/codex.jsonl"),
            0,
            payload,
            &BackfillParseContext::default(),
        )
        .unwrap();
        let second = append_prepared_event(&home, backfill_event).unwrap();

        let raw =
            fs::read_to_string(canonical_raw_path(&home, Tool::Codex, "fixture-session")).unwrap();
        assert!(first.appended);
        assert!(!second.appended);
        assert_eq!(raw.lines().count(), 1);
        index_once(&home).unwrap();
        let results = search_history(&home, "cross source dedupe fixture marker", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].session_id, "fixture-session");
    }

    #[test]
    fn raw_append_dedupes_unsequenced_event_across_observation_time_and_route() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();
        let hook_payload = json!({
            "session_id": "fixture-session",
            "hook_event_name": "UserPromptSubmit",
            "captured_at": "2026-06-17T12:00:59Z",
            "cwd": "/tmp/nabu-fixture",
            "project_root": "/tmp/nabu-fixture",
            "prompt": "unsequenced duplicate marker"
        });
        let backfill_payload = json!({
            "session_id": "fixture-session",
            "hook_event_name": "UserPromptSubmit",
            "captured_at": "2026-06-17T12:01:01Z",
            "cwd": "/tmp/nabu-fixture",
            "project_root": "/tmp/nabu-fixture",
            "prompt": "unsequenced duplicate marker"
        });

        let first = ingest_hook_event(&home, Tool::Claude, hook_payload).unwrap();
        let mut event = envelope_from_backfill_payload(
            Tool::Claude,
            Path::new("/tmp/claude-transcript.jsonl"),
            42,
            backfill_payload,
            &BackfillParseContext::default(),
        )
        .unwrap();
        event.sequence = None;
        let second = append_prepared_event(&home, event).unwrap();

        let raw =
            fs::read_to_string(canonical_raw_path(&home, Tool::Claude, "fixture-session")).unwrap();
        assert!(first.appended);
        assert!(!second.appended);
        assert_eq!(raw.lines().count(), 1);

        index_once(&home).unwrap();
        let results = search_history(&home, "unsequenced duplicate marker", 10).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn dedupe_sidecar_covers_large_session_and_self_heals() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();
        let session_id = "large-sidecar-session";
        let seed_events = (0..10_000)
            .map(|index| {
                envelope_from_backfill_payload(
                    Tool::Claude,
                    Path::new("/tmp/large-sidecar.jsonl"),
                    index as u64,
                    json!({
                        "session_id": session_id,
                        "hook_event_name": "UserPromptSubmit",
                        "message_id": format!("large-sidecar-{index}"),
                        "sequence": index as i64,
                        "cwd": "/tmp/nabu-fixture",
                        "project_root": "/tmp/nabu-fixture",
                        "prompt": format!("large sidecar marker {index}")
                    }),
                    &BackfillParseContext::default(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        append_prepared_events(&home, seed_events).unwrap();

        let raw_path = canonical_raw_path(&home, Tool::Claude, session_id);
        let sidecar = DedupeSidecarFiles::for_raw_file(&home, &raw_path);
        assert_eq!(raw_line_count(&raw_path), 10_000);
        assert_eq!(dedupe_sidecar_entry_count(&sidecar), 10_000);

        let duplicate = ingest_hook_event(
            &home,
            Tool::Claude,
            json!({
                "session_id": session_id,
                "hook_event_name": "UserPromptSubmit",
                "message_id": "large-sidecar-1234",
                "sequence": 1234,
                "cwd": "/tmp/nabu-fixture",
                "project_root": "/tmp/nabu-fixture",
                "prompt": "large sidecar marker 1234"
            }),
        )
        .unwrap();
        assert!(!duplicate.appended);
        assert_eq!(duplicate.raw_offset, raw_offset_for_line(&raw_path, 1234));
        assert_eq!(raw_line_count(&raw_path), 10_000);

        fs::remove_dir_all(&sidecar.buckets_dir).unwrap();
        let duplicate_after_delete = ingest_hook_event(
            &home,
            Tool::Claude,
            json!({
                "session_id": session_id,
                "hook_event_name": "UserPromptSubmit",
                "message_id": "large-sidecar-4321",
                "sequence": 4321,
                "cwd": "/tmp/nabu-fixture",
                "project_root": "/tmp/nabu-fixture",
                "prompt": "large sidecar marker 4321"
            }),
        )
        .unwrap();
        assert!(!duplicate_after_delete.appended);
        assert_eq!(raw_line_count(&raw_path), 10_000);
        assert_eq!(dedupe_sidecar_entry_count(&sidecar), 10_000);

        let corrupt_payload = json!({
            "session_id": session_id,
            "hook_event_name": "UserPromptSubmit",
            "message_id": "large-sidecar-9876",
            "sequence": 9876,
            "cwd": "/tmp/nabu-fixture",
            "project_root": "/tmp/nabu-fixture",
            "prompt": "large sidecar marker 9876"
        });
        let corrupt_event = envelope_from_backfill_payload(
            Tool::Claude,
            Path::new("/tmp/large-sidecar.jsonl"),
            9876,
            corrupt_payload.clone(),
            &BackfillParseContext::default(),
        )
        .unwrap();
        let corrupt_key = dedupe_key(DedupeParts {
            tool: corrupt_event.tool,
            session_id: &corrupt_event.session_id,
            canonical_type: corrupt_event.canonical_type,
            source_event_id: corrupt_event.source_event_id.as_deref(),
            sequence: corrupt_event.sequence,
            payload: &corrupt_event.payload,
        })
        .unwrap();
        let corrupt_bucket = dedupe_bucket_index(&corrupt_key).unwrap();
        fs::write(sidecar.bucket_path(corrupt_bucket), b"sha256:truncated").unwrap();
        let duplicate_after_corruption =
            ingest_hook_event(&home, Tool::Claude, corrupt_payload).unwrap();
        assert!(!duplicate_after_corruption.appended);
        assert_eq!(raw_line_count(&raw_path), 10_000);
        assert_eq!(dedupe_sidecar_entry_count(&sidecar), 10_000);

        index_once(&home).unwrap();
        let results = search_history(&home, "large sidecar marker 1234", 10).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn concurrent_appends_keep_raw_and_sidecar_consistent() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();
        let session_id = "concurrent-sidecar-session";
        let mut handles = Vec::new();

        for index in 0..64 {
            let home = home.clone();
            handles.push(std::thread::spawn(move || {
                ingest_hook_event(
                    &home,
                    Tool::Codex,
                    json!({
                        "session_id": session_id,
                        "hook_event_name": "UserPromptSubmit",
                        "message_id": format!("concurrent-sidecar-{index}"),
                        "sequence": index as i64,
                        "cwd": "/tmp/nabu-fixture",
                        "project_root": "/tmp/nabu-fixture",
                        "prompt": format!("concurrent sidecar marker {index}")
                    }),
                )
                .unwrap()
            }));
        }

        for handle in handles {
            assert!(handle.join().unwrap().appended);
        }

        let raw_path = canonical_raw_path(&home, Tool::Codex, session_id);
        let sidecar = DedupeSidecarFiles::for_raw_file(&home, &raw_path);
        assert_eq!(raw_line_count(&raw_path), 64);
        assert_eq!(dedupe_sidecar_entry_count(&sidecar), 64);
    }

    #[test]
    fn native_order_preserves_identical_content_and_unordered_still_collapses() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();

        for sequence in [1, 2] {
            let report = ingest_hook_event(
                &home,
                Tool::Codex,
                json!({
                    "session_id": "ordered-identical-session",
                    "hook_event_name": "UserPromptSubmit",
                    "sequence": sequence,
                    "cwd": "/tmp/nabu-fixture",
                    "project_root": "/tmp/nabu-fixture",
                    "prompt": "identical ordered content marker"
                }),
            )
            .unwrap();
            assert!(report.appended);
        }
        assert_eq!(
            raw_line_count(&canonical_raw_path(
                &home,
                Tool::Codex,
                "ordered-identical-session"
            )),
            2
        );

        let first = ingest_hook_event(
            &home,
            Tool::Codex,
            json!({
                "session_id": "unordered-identical-session",
                "hook_event_name": "UserPromptSubmit",
                "cwd": "/tmp/nabu-fixture",
                "project_root": "/tmp/nabu-fixture",
                "prompt": "identical unordered content marker"
            }),
        )
        .unwrap();
        let second = ingest_hook_event(
            &home,
            Tool::Codex,
            json!({
                "session_id": "unordered-identical-session",
                "hook_event_name": "UserPromptSubmit",
                "captured_at": "2099-01-01T00:00:00Z",
                "cwd": "/tmp/nabu-fixture",
                "project_root": "/tmp/nabu-fixture",
                "prompt": "identical unordered content marker"
            }),
        )
        .unwrap();
        assert!(first.appended);
        assert!(!second.appended);
        assert_eq!(
            raw_line_count(&canonical_raw_path(
                &home,
                Tool::Codex,
                "unordered-identical-session"
            )),
            1
        );
    }

    #[test]
    fn source_specific_ordering_fields_are_mapped_to_sequence() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();

        for part_index in [7, 8] {
            let report = ingest_hook_event(
                &home,
                Tool::Opencode,
                json!({
                    "session_id": "opencode-part-order-session",
                    "hook_event_name": "message.part.updated",
                    "message_id": "shared-opencode-message",
                    "part": {
                        "index": part_index,
                        "text": "same opencode part text marker"
                    },
                    "cwd": "/tmp/nabu-fixture",
                    "project_root": "/tmp/nabu-fixture",
                    "delta": "same opencode part text marker"
                }),
            )
            .unwrap();
            assert!(report.appended);
        }

        for item_index in [3, 4] {
            let report = ingest_hook_event(
                &home,
                Tool::Codex,
                json!({
                    "session_id": "codex-item-order-session",
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "item_index": item_index,
                        "content": [{"type": "output_text", "text": "same codex item text marker"}]
                    },
                    "cwd": "/tmp/nabu-fixture",
                    "project_root": "/tmp/nabu-fixture"
                }),
            )
            .unwrap();
            assert!(report.appended);
        }

        let first = envelope_from_backfill_payload(
            Tool::Claude,
            Path::new("/tmp/transcript.jsonl"),
            10,
            json!({
                "session_id": "backfill-offset-order-session",
                "hook_event_name": "UserPromptSubmit",
                "cwd": "/tmp/nabu-fixture",
                "project_root": "/tmp/nabu-fixture",
                "prompt": "same backfill offset text marker"
            }),
            &BackfillParseContext::default(),
        )
        .unwrap();
        let second = envelope_from_backfill_payload(
            Tool::Claude,
            Path::new("/tmp/transcript.jsonl"),
            20,
            json!({
                "session_id": "backfill-offset-order-session",
                "hook_event_name": "UserPromptSubmit",
                "cwd": "/tmp/nabu-fixture",
                "project_root": "/tmp/nabu-fixture",
                "prompt": "same backfill offset text marker"
            }),
            &BackfillParseContext::default(),
        )
        .unwrap();
        assert!(append_prepared_event(&home, first).unwrap().appended);
        assert!(append_prepared_event(&home, second).unwrap().appended);

        assert_eq!(
            raw_line_count(&canonical_raw_path(
                &home,
                Tool::Opencode,
                "opencode-part-order-session"
            )),
            2
        );
        assert_eq!(
            raw_line_count(&canonical_raw_path(
                &home,
                Tool::Codex,
                "codex-item-order-session"
            )),
            2
        );
        assert_eq!(
            raw_line_count(&canonical_raw_path(
                &home,
                Tool::Claude,
                "backfill-offset-order-session"
            )),
            2
        );
    }

    #[test]
    fn codex_native_transcript_backfill_derives_session_id_from_metadata() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        let source = temp.path().join("codex-sessions");
        init_home(&home).unwrap();
        fs::create_dir_all(&source).unwrap();

        let session_id = "019a4b44-cc3b-7c51-8944-a7d7ebb9e6fe";
        fs::write(
            source.join(format!("rollout-2025-11-03T20-49-51-{session_id}.jsonl")),
            format!(
                "{{\"timestamp\":\"2025-11-03T19:49:51.304Z\",\"type\":\"session_meta\",\"payload\":{{\"id\":\"{session_id}\",\"cwd\":\"/tmp/native-codex\"}}}}\n\
                 {{\"timestamp\":\"2025-11-03T19:50:01.966Z\",\"type\":\"response_item\",\"payload\":{{\"type\":\"message\",\"role\":\"user\",\"content\":[{{\"type\":\"input_text\",\"text\":\"native codex backfill marker\"}}]}}}}\n"
            ),
        )
        .unwrap();

        let report = backfill_since(&home, Some(Tool::Codex), &source, None).unwrap();
        assert_eq!(report.source_files, 1);
        assert_eq!(report.appended_events, 2);
        assert_eq!(report.checkpoint_files, 1);

        index_once(&home).unwrap();
        let results = search_history(&home, "native codex backfill marker", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].tool, Tool::Codex);
        assert_eq!(results[0].session_id, session_id);
        assert_eq!(results[0].canonical_type, "user.message");
    }

    #[cfg(unix)]
    #[test]
    fn backfill_skips_source_file_that_vanishes_before_read() {
        // A session file discovered during the scan can be deleted/rotated by the
        // live tool before backfill reads it (os error 2). One vanished file must
        // not abort the whole backfill. A dangling symlink reproduces a candidate
        // that passes discovery but fails with NotFound on read.
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        let source = temp.path().join("codex-sessions");
        init_home(&home).unwrap();
        fs::create_dir_all(&source).unwrap();

        let session_id = "019a4f57-3d5f-7f52-96cc-cb2e1eacb7a9";
        fs::write(
            source.join(format!("rollout-2025-11-04T15-48-28-{session_id}.jsonl")),
            "{\"timestamp\":\"2025-11-04T14:48:28.000Z\",\"type\":\"response_item\",\"payload\":{\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"surviving codex marker\"}]}}\n",
        )
        .unwrap();
        // Discovered by extension, but reading it yields NotFound.
        std::os::unix::fs::symlink(
            temp.path().join("does-not-exist.jsonl"),
            source.join("rollout-2025-11-04T16-00-00-vanished.jsonl"),
        )
        .unwrap();

        // Dry run (the wizard's "Scanning past sessions…") must not fail.
        let dry = backfill_dry_run(&home, Some(Tool::Codex), &source, None).unwrap();
        assert_eq!(dry.source_files, 1);

        // The real backfill must skip the vanished file and import the valid one.
        let report = backfill_since(&home, Some(Tool::Codex), &source, None).unwrap();
        assert_eq!(report.source_files, 1);
        assert_eq!(report.appended_events, 1);

        index_once(&home).unwrap();
        let results = search_history(&home, "surviving codex marker", 10).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn codex_native_transcript_backfill_derives_session_id_from_filename() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        let source = temp.path().join("codex-sessions");
        init_home(&home).unwrap();
        fs::create_dir_all(&source).unwrap();

        let session_id = "019a4f57-3d5f-7f52-96cc-cb2e1eacb7a9";
        fs::write(
            source.join(format!("rollout-2025-11-04T15-48-28-{session_id}.jsonl")),
            "{\"timestamp\":\"2025-11-04T14:48:28.000Z\",\"type\":\"response_item\",\"payload\":{\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"filename derived codex marker\"}]}}\n",
        )
        .unwrap();

        let report = backfill_since(&home, Some(Tool::Codex), &source, None).unwrap();
        assert_eq!(report.appended_events, 1);

        index_once(&home).unwrap();
        let results = search_history(&home, "filename derived codex marker", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].session_id, session_id);
    }

    #[test]
    fn claude_native_backfill_ignores_project_sidecars() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        let source = temp.path().join("claude-projects");
        init_home(&home).unwrap();
        fs::create_dir_all(source.join("session/subagents")).unwrap();

        let session_id = "15a53bfd-c382-488d-b890-687021285e49";
        fs::write(
            source.join(format!("{session_id}.jsonl")),
            format!(
                "{{\"session_id\":\"{session_id}\",\"cwd\":\"/tmp/native-claude\",\"project_root\":\"/tmp/native-claude\",\"type\":\"claude.transcript.user\",\"canonical_type\":\"user.message\",\"event_id\":\"claude-native-1\",\"message\":\"native claude marker\"}}\n"
            ),
        )
        .unwrap();
        fs::write(source.join("sessions-index.json"), "{\"sessions\":[]}").unwrap();
        fs::write(
            source.join("session/subagents/agent-a676598cc8f883f73.meta.json"),
            "{\"agent_id\":\"agent-a676598cc8f883f73\"}",
        )
        .unwrap();
        fs::write(
            source.join("session/subagents/skill-injections.jsonl"),
            "{\"kind\":\"plugin-config\",\"type\":\"skill-injection\"}\n",
        )
        .unwrap();

        let report = backfill_since(&home, Some(Tool::Claude), &source, None).unwrap();
        assert_eq!(report.source_files, 1);
        assert_eq!(report.appended_events, 1);
        assert_eq!(report.checkpoint_files, 1);

        index_once(&home).unwrap();
        let results = search_history(&home, "native claude marker", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].session_id, session_id);
    }

    #[test]
    fn sanitized_real_native_fixtures_import_defensively_for_all_tools() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        init_home(&home).unwrap();

        let claude_report = backfill_since(
            &home,
            Some(Tool::Claude),
            &repo.join("fixtures/native/claude/projects"),
            None,
        )
        .unwrap();
        let codex_report = backfill_since(
            &home,
            Some(Tool::Codex),
            &repo.join("fixtures/native/codex/sessions"),
            None,
        )
        .unwrap();
        let opencode_report = backfill_since(
            &home,
            Some(Tool::Opencode),
            &repo.join("fixtures/native/opencode"),
            None,
        )
        .unwrap();

        assert_eq!(claude_report.source_files, 1);
        assert_eq!(claude_report.appended_events, 5);
        assert_eq!(codex_report.source_files, 1);
        assert_eq!(codex_report.appended_events, 4);
        assert_eq!(opencode_report.source_files, 5);
        assert_eq!(opencode_report.appended_events, 8);
        assert_eq!(checkpoint_row_count(&home), 7);

        index_once(&home).unwrap();
        assert_eq!(
            search_history(&home, "sanitized native claude user marker", 10).unwrap()[0].session_id,
            "11111111-1111-4111-8111-111111111111"
        );
        assert_eq!(
            search_history(&home, "sanitized native codex assistant marker", 10).unwrap()[0]
                .session_id,
            "22222222-2222-4222-8222-222222222222"
        );
        assert_eq!(
            search_history(&home, "sanitized native opencode assistant marker", 10).unwrap()[0]
                .session_id,
            "33333333-3333-4333-8333-333333333333"
        );

        let claude_raw =
            canonical_raw_path(&home, Tool::Claude, "11111111-1111-4111-8111-111111111111");
        let envelopes = raw_envelopes(&claude_raw);
        let parse_error = envelopes
            .iter()
            .find(|event| event.canonical_type == CanonicalType::Error)
            .expect("malformed native line should import as error");
        assert_eq!(
            parse_error.payload.get("type").and_then(Value::as_str),
            Some("parse_error")
        );
        assert!(parse_error.payload.get("raw_line").is_some());
    }

    #[test]
    fn opencode_native_fixture_maps_m8_types_worktree_and_metadata_session() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let fixture = repo.join("fixtures/native/opencode");
        let session_id = "33333333-3333-4333-8333-333333333333";
        init_home(&home).unwrap();

        let first = backfill_since(&home, Some(Tool::Opencode), &fixture, None).unwrap();
        let second = backfill_since(&home, Some(Tool::Opencode), &fixture, None).unwrap();

        assert_eq!(first.source_files, 5);
        assert_eq!(first.appended_events, 8);
        assert_eq!(second.appended_events, 0);

        let raw_path = canonical_raw_path(&home, Tool::Opencode, session_id);
        let envelopes = raw_envelopes(&raw_path);
        assert_eq!(envelopes.len(), 8);
        let error_count = envelopes
            .iter()
            .filter(|event| event.canonical_type == CanonicalType::Error)
            .count();
        assert_eq!(error_count, 0);

        assert!(envelopes.iter().any(|event| {
            event.source_event_type == "reasoning"
                && event.canonical_type == CanonicalType::AssistantDelta
        }));
        assert!(envelopes.iter().any(|event| {
            event.source_event_type == "step-start"
                && event.canonical_type == CanonicalType::AssistantDelta
        }));
        assert!(envelopes.iter().any(|event| {
            event.source_event_type == "step-finish"
                && event.canonical_type == CanonicalType::AssistantDelta
        }));
        assert!(envelopes.iter().any(|event| {
            event.source_event_type == "patch" && event.canonical_type == CanonicalType::FileChanged
        }));
        assert!(envelopes.iter().any(|event| {
            event.source_event_type == "session.created"
                && event.canonical_type == CanonicalType::SessionStarted
        }));
        assert!(envelopes
            .iter()
            .all(|event| event.project_root.is_some() && event.cwd.is_some()));
        assert!(envelopes.iter().any(|event| {
            event.source_event_type == "reasoning"
                && event.project_root.as_deref() == Some("/Users/example/opencode-project")
                && event.cwd.as_deref() == Some("/Users/example/opencode-project")
        }));
        assert!(!canonical_raw_path(&home, Tool::Opencode, "project_meta").exists());
    }

    #[test]
    fn backfill_uses_sqlite_checkpoints_and_incremental_rerun_is_noop() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        let source = temp.path().join("codex-sessions");
        let session_id = "44444444-4444-4444-8444-444444444444";
        init_home(&home).unwrap();
        fs::create_dir_all(&source).unwrap();
        fs::write(
            source.join(format!("rollout-2026-06-18T10-00-00-{session_id}.jsonl")),
            format!(
                "{{\"timestamp\":\"2026-06-18T10:00:00.000Z\",\"type\":\"session_meta\",\"payload\":{{\"id\":\"{session_id}\",\"cwd\":\"/tmp/native-codex\"}}}}\n\
                 {{\"timestamp\":\"2026-06-18T10:00:01.000Z\",\"type\":\"response_item\",\"payload\":{{\"id\":\"checkpoint-user-1\",\"type\":\"message\",\"role\":\"user\",\"content\":[{{\"type\":\"input_text\",\"text\":\"checkpoint native marker\"}}]}}}}\n"
            ),
        )
        .unwrap();

        let first = backfill_since(&home, Some(Tool::Codex), &source, None).unwrap();
        let second = backfill_since(&home, Some(Tool::Codex), &source, None).unwrap();

        assert_eq!(first.appended_events, 2);
        assert_eq!(first.checkpoint_files, 1);
        assert_eq!(second.appended_events, 0);
        assert_eq!(checkpoint_row_count(&home), 1);
        assert_eq!(checkpoint_sidecar_count(&home), 0);
        assert_eq!(
            raw_line_count(&canonical_raw_path(&home, Tool::Codex, session_id)),
            2
        );
    }

    #[test]
    fn raw_index_checkpoints_skip_unchanged_canonical_files_and_refresh_on_append() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        let session_id = "raw-index-checkpoint-session";
        init_home(&home).unwrap();
        ingest_hook_event(
            &home,
            Tool::Claude,
            json!({
                "session_id": session_id,
                "hook_event_name": "UserPromptSubmit",
                "message_id": "raw-index-checkpoint-1",
                "prompt": "raw checkpoint first marker"
            }),
        )
        .unwrap();

        let raw_path = canonical_raw_path(&home, Tool::Claude, session_id);
        let first = index_once(&home).unwrap();
        let second = index_once(&home).unwrap();
        let db_path = home.join("index").join("harness.db");
        let conn = open_index(&db_path).unwrap();
        let source_meta = source_file_metadata(&raw_path).unwrap();

        assert_eq!(first.indexed_events, 1);
        assert_eq!(second.indexed_events, 0);
        assert!(raw_index_checkpoint_is_current(
            &conn,
            &db_path,
            Tool::Claude,
            &raw_path,
            &source_meta
        )
        .unwrap());
        let raw_checkpoint_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM checkpoints WHERE source_kind = 'raw_jsonl'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(raw_checkpoint_count, 1);

        ingest_hook_event(
            &home,
            Tool::Claude,
            json!({
                "session_id": session_id,
                "hook_event_name": "UserPromptSubmit",
                "message_id": "raw-index-checkpoint-2",
                "prompt": "raw checkpoint second marker"
            }),
        )
        .unwrap();
        let changed_meta = source_file_metadata(&raw_path).unwrap();
        assert!(!raw_index_checkpoint_is_current(
            &conn,
            &db_path,
            Tool::Claude,
            &raw_path,
            &changed_meta
        )
        .unwrap());

        let third = index_once(&home).unwrap();
        assert_eq!(third.indexed_events, 1);
    }

    #[test]
    fn fts_schema_migration_rebuilds_without_reindexing_raw_files() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();
        ingest_hook_event(
            &home,
            Tool::Claude,
            json!({
                "session_id": "fts-migration-session",
                "hook_event_name": "UserPromptSubmit",
                "message_id": "fts-migration-1",
                "prompt": "streaming fts rebuild marker"
            }),
        )
        .unwrap();

        assert_eq!(index_once(&home).unwrap().indexed_events, 1);
        let db_path = home.join("index").join("harness.db");
        Connection::open(&db_path)
            .unwrap()
            .execute_batch(
                "DROP TABLE IF EXISTS events_fts;
                 CREATE VIRTUAL TABLE events_fts USING fts5(searchable_text);",
            )
            .unwrap();

        assert_eq!(index_once(&home).unwrap().indexed_events, 0);
        let results = search_history(&home, "streaming fts rebuild marker", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].session_id, "fts-migration-session");
    }

    #[test]
    fn discontinuities_emit_once_for_truncation_rotation_and_deletion() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        let root = temp.path().join("claude-projects");
        init_home(&home).unwrap();
        fs::create_dir_all(&root).unwrap();

        let truncated_session = "55555555-5555-4555-8555-555555555555";
        let truncated = root.join(format!("{truncated_session}.jsonl"));
        fs::write(
            &truncated,
            claude_user_line(truncated_session, "truncate-1", "truncation original one")
                + &claude_user_line(truncated_session, "truncate-2", "truncation original two"),
        )
        .unwrap();
        backfill_since(&home, Some(Tool::Claude), &root, None).unwrap();
        fs::write(
            &truncated,
            claude_user_line(truncated_session, "truncate-1", "truncation original one"),
        )
        .unwrap();
        assert_eq!(
            backfill_since(&home, Some(Tool::Claude), &root, None)
                .unwrap()
                .discontinuities,
            1
        );
        assert_eq!(
            backfill_since(&home, Some(Tool::Claude), &root, None)
                .unwrap()
                .discontinuities,
            0
        );
        assert_eq!(
            discontinuity_count(&home, Tool::Claude, truncated_session, "source.truncated"),
            1
        );

        let rotated_session = "66666666-6666-4666-8666-666666666666";
        let rotated = root.join(format!("{rotated_session}.jsonl"));
        fs::write(
            &rotated,
            claude_user_line(rotated_session, "rotate-1", "rotation original marker"),
        )
        .unwrap();
        backfill_since(&home, Some(Tool::Claude), &root, None).unwrap();
        fs::remove_file(&rotated).unwrap();
        fs::write(
            &rotated,
            claude_user_line(
                rotated_session,
                "rotate-2",
                "rotation replacement marker with enough bytes to avoid truncation precedence",
            ),
        )
        .unwrap();
        assert_eq!(
            backfill_since(&home, Some(Tool::Claude), &root, None)
                .unwrap()
                .discontinuities,
            1
        );
        assert_eq!(
            backfill_since(&home, Some(Tool::Claude), &root, None)
                .unwrap()
                .discontinuities,
            0
        );
        assert_eq!(
            discontinuity_count(&home, Tool::Claude, rotated_session, "source.rotated"),
            1
        );

        let deleted_session = "77777777-7777-4777-8777-777777777777";
        let deleted = root.join(format!("{deleted_session}.jsonl"));
        fs::write(
            &deleted,
            claude_user_line(deleted_session, "delete-1", "deletion original marker"),
        )
        .unwrap();
        backfill_since(&home, Some(Tool::Claude), &root, None).unwrap();
        fs::remove_file(&deleted).unwrap();
        assert_eq!(
            backfill_since(&home, Some(Tool::Claude), &root, None)
                .unwrap()
                .discontinuities,
            1
        );
        assert_eq!(
            backfill_since(&home, Some(Tool::Claude), &root, None)
                .unwrap()
                .discontinuities,
            0
        );
        assert_eq!(
            discontinuity_count(&home, Tool::Claude, deleted_session, "source.deleted"),
            1
        );
    }

    #[test]
    fn dry_run_reports_missing_events_and_writes_nothing() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        let source = temp.path().join("codex-sessions");
        let session_id = "88888888-8888-4888-8888-888888888888";
        init_home(&home).unwrap();
        fs::create_dir_all(&source).unwrap();
        fs::write(
            source.join(format!("rollout-2026-06-18T11-00-00-{session_id}.jsonl")),
            "{\"timestamp\":\"2026-06-18T11:00:00.000Z\",\"type\":\"response_item\",\"payload\":{\"id\":\"dry-run-shared\",\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"dry run shared marker\"}]}}\n\
                 {\"timestamp\":\"2026-06-18T11:00:01.000Z\",\"type\":\"response_item\",\"payload\":{\"id\":\"dry-run-gap\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"dry run gap marker\"}]}}\n",
        )
        .unwrap();
        ingest_hook_event(
            &home,
            Tool::Codex,
            json!({
                "session_id": session_id,
                "type": "response_item",
                "payload": {
                    "id": "dry-run-shared",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "dry run shared marker"}]
                }
            }),
        )
        .unwrap();

        let before_raw_lines = raw_line_count(&canonical_raw_path(&home, Tool::Codex, session_id));
        let report = backfill_dry_run(&home, Some(Tool::Codex), &source, None).unwrap();
        let after_raw_lines = raw_line_count(&canonical_raw_path(&home, Tool::Codex, session_id));

        assert_eq!(report.source_files, 1);
        assert_eq!(report.on_disk_events, 2);
        assert_eq!(report.captured_events, 1);
        assert_eq!(report.missing_events, 1);
        assert_eq!(report.partial_sessions, 1);
        assert_eq!(report.sessions[0].would_import.len(), 1);
        assert_eq!(
            report.sessions[0].would_import[0].canonical_type,
            "assistant.message"
        );
        assert_eq!(before_raw_lines, after_raw_lines);
        assert_eq!(checkpoint_row_count(&home), 0);
    }

    #[test]
    fn partial_live_capture_then_backfill_reconciles_for_each_tool() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();

        assert_reconciles_claude(&home, temp.path());
        assert_reconciles_codex(&home, temp.path());
        assert_reconciles_opencode(&home, temp.path());
    }

    #[test]
    fn doctor_reports_compact_coverage_summary() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        let source = temp.path().join("claude-projects");
        let session_id = "99999999-9999-4999-8999-999999999999";
        init_home(&home).unwrap();
        fs::create_dir_all(&source).unwrap();
        fs::write(
            source.join(format!("{session_id}.jsonl")),
            claude_user_line(session_id, "doctor-coverage-1", "doctor coverage marker"),
        )
        .unwrap();
        backfill_since(&home, Some(Tool::Claude), &source, None).unwrap();
        index_once(&home).unwrap();

        let report = doctor_with_options(&home, false);

        assert_eq!(report.coverage.checkpointed_sources, 2);
        assert_eq!(report.coverage.captured_sessions, 1);
        assert_eq!(report.coverage.captured_events, 1);
        assert!(report.storage_footprint.raw_bytes > 0);
        assert!(report.storage_footprint.index_bytes > 0);
        assert_eq!(report.storage_footprint.vectors_bytes, 0);
        assert_eq!(report.storage_footprint.models_bytes, 0);
        assert_eq!(
            report.storage_footprint.canonical_total,
            report
                .storage_footprint
                .raw_bytes
                .saturating_add(report.storage_footprint.blobs_bytes)
        );
        assert_eq!(
            report.storage_footprint.derived_total,
            report
                .storage_footprint
                .index_bytes
                .saturating_add(report.storage_footprint.spool_bytes)
                .saturating_add(report.storage_footprint.models_bytes)
        );
        assert!(report.storage_footprint.total_bytes >= report.storage_footprint.raw_bytes);
    }

    #[cfg(not(feature = "semantic"))]
    #[test]
    fn default_build_reports_no_semantic_model_without_touching_network() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();

        let status = embedding_model_status(&home);

        assert!(!status.feature_enabled);
        assert_eq!(status.model_id, "embeddinggemma-300m-q4");
        assert_eq!(status.expected_dimensions, 256);
        assert!(!status.model_present);
        assert!(!status.semantic_available);
        assert!(status.message.contains("semantic feature is disabled"));
    }

    #[test]
    fn embedding_model_disclosure_reports_terms_and_measured_local_footprint() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();
        let cache_path = semantic_model_cache_path(&home);
        fs::create_dir_all(&cache_path).unwrap();
        fs::write(cache_path.join("partial-file.bin"), b"partial model bytes").unwrap();

        let disclosure = embedding_model_disclosure(&home, SEMANTIC_MODEL_ID).unwrap();

        assert_eq!(disclosure.model_id, SEMANTIC_MODEL_ID);
        assert_eq!(disclosure.repository, SEMANTIC_MODEL_REPO);
        assert_eq!(disclosure.total_files, SEMANTIC_MODEL_REMOTE_FILES.len());
        assert!(disclosure.current_on_disk_bytes >= "partial model bytes".len() as u64);
        assert!(!disclosure.model_present);
        assert!(disclosure.license_summary.contains("Gemma Terms of Use"));
        assert!(disclosure.cache_path.ends_with(SEMANTIC_MODEL_ID));
    }

    #[test]
    fn semantic_retrieval_fixture_is_labeled_without_requiring_model() {
        let fixture = semantic_retrieval_fixture();
        assert_eq!(fixture.schema_version, 1);
        assert_eq!(fixture.tool, Tool::Claude);
        assert!(!fixture.session_id.trim().is_empty());
        assert!(!fixture.cwd.trim().is_empty());
        assert!(!fixture.project_root.trim().is_empty());
        assert!(!fixture.events.is_empty());
        assert!(!fixture.queries.is_empty());

        let event_ids = fixture
            .events
            .iter()
            .map(|event| {
                assert!(!event.event_id.trim().is_empty());
                assert!(matches!(event.role.as_str(), "user" | "assistant"));
                assert!(!event.text.trim().is_empty());
                event.event_id.clone()
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(event_ids.len(), fixture.events.len());

        for query in &fixture.queries {
            assert!(!query.query.trim().is_empty());
            assert!(!query.relevant_event_ids.is_empty());
            for event_id in &query.relevant_event_ids {
                assert!(
                    event_ids.contains(event_id),
                    "query {:?} references unknown event id {event_id}",
                    query.query
                );
            }
        }
    }

    #[test]
    fn corroboration_extracts_and_resolves_refs_read_only_against_local_git() {
        let extraction_refs = extract_corroboration_candidates(
            "commit abcdef1 landed on branch feature/corroborate, touched src/lib.rs, and referenced PR #42.",
        )
        .into_iter()
        .map(|candidate| (candidate.kind.as_str().to_string(), candidate.reference))
        .collect::<BTreeSet<_>>();
        assert_eq!(
            extraction_refs,
            BTreeSet::from([
                ("branch".to_string(), "feature/corroborate".to_string()),
                ("commit".to_string(), "abcdef1".to_string()),
                ("file".to_string(), "src/lib.rs".to_string()),
                ("pr".to_string(), "#42".to_string()),
            ])
        );

        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        let repo = temp.path().join("repo");
        fs::create_dir_all(repo.join("src")).unwrap();
        run_git_setup(temp.path(), &["init", repo.to_str().unwrap()]);
        run_git(&repo, &["config", "user.email", "nabu@example.invalid"]);
        run_git(&repo, &["config", "user.name", "Nabu Test"]);
        fs::write(repo.join("src/lib.rs"), "pub fn corroborated() {}\n").unwrap();
        run_git(&repo, &["add", "src/lib.rs"]);
        run_git(&repo, &["commit", "-m", "initial corroboration fixture"]);
        run_git(&repo, &["branch", "feature/corroborate"]);
        fs::create_dir_all(repo.join("notes")).unwrap();
        fs::write(
            repo.join("notes/trace.txt"),
            "untracked corroboration note\n",
        )
        .unwrap();
        let commit = run_git(&repo, &["rev-parse", "HEAD"]);
        let commit_prefix = &commit[..12];
        let missing_commit = "ffffffffffffffffffffffffffffffffffffffff";
        let before_snapshot = git_snapshot(&repo);

        init_home(&home).unwrap();
        let text = format!(
            "corroboration marker commit {commit_prefix} and missing commit {missing_commit}; branch feature/corroborate and branch missing/branch; files src/lib.rs notes/trace.txt src/missing.txt; PR #123."
        );
        ingest_hook_event(
            &home,
            Tool::Claude,
            json!({
                "session_id": "corroboration-session",
                "hook_event_name": "UserPromptSubmit",
                "message_id": "corroboration-message",
                "cwd": repo,
                "project_root": repo,
                "prompt": text,
            }),
        )
        .unwrap();
        index_once(&home).unwrap();

        clear_git_invocations();
        let default_page = search_history_page(
            &home,
            "corroboration marker",
            SearchOptions {
                limit: 1,
                ..SearchOptions::default()
            },
        )
        .unwrap();
        assert_eq!(default_page.returned, 1);
        assert!(default_page.results[0].corroboration.is_none());
        assert!(captured_git_invocations().is_empty());

        let page = search_history_page(
            &home,
            "corroboration marker",
            SearchOptions {
                limit: 1,
                corroborate: true,
                ..SearchOptions::default()
            },
        )
        .unwrap();
        let corroboration = page.results[0].corroboration.as_ref().unwrap();
        let canonical_repo = fs::canonicalize(&repo).unwrap();
        assert_eq!(
            corroboration.repo.as_deref(),
            Some(canonical_repo.to_str().unwrap())
        );
        assert_ref_status(corroboration, "commit", commit_prefix, "present", None);
        assert_ref_status(corroboration, "commit", missing_commit, "missing", None);
        assert_ref_status(
            corroboration,
            "branch",
            "feature/corroborate",
            "present",
            None,
        );
        assert_ref_status(corroboration, "branch", "missing/branch", "missing", None);
        assert_ref_status(corroboration, "file", "src/lib.rs", "present", None);
        assert_ref_status(corroboration, "file", "notes/trace.txt", "untracked", None);
        assert_ref_status(corroboration, "file", "src/missing.txt", "missing", None);
        assert_ref_status(
            corroboration,
            "pr",
            "#123",
            "unresolved",
            Some("needs_network"),
        );
        assert_eq!(git_snapshot(&repo), before_snapshot);
        assert_no_network_git_commands(&captured_git_invocations());

        let no_repo = temp.path().join("no-repo");
        fs::create_dir_all(&no_repo).unwrap();
        ingest_hook_event(
            &home,
            Tool::Claude,
            json!({
                "session_id": "no-repo-corroboration-session",
                "hook_event_name": "UserPromptSubmit",
                "message_id": "no-repo-corroboration-message",
                "cwd": no_repo,
                "project_root": no_repo,
                "prompt": "no repo marker commit deadbee file src/lib.rs PR #7",
            }),
        )
        .unwrap();
        index_once(&home).unwrap();
        let no_repo_page = search_history_page(
            &home,
            "no repo marker",
            SearchOptions {
                limit: 1,
                corroborate: true,
                ..SearchOptions::default()
            },
        )
        .unwrap();
        let no_repo_corroboration = no_repo_page.results[0].corroboration.as_ref().unwrap();
        assert_eq!(no_repo_corroboration.repo, None);
        assert_ref_status(
            no_repo_corroboration,
            "commit",
            "deadbee",
            "unresolved",
            Some("no_repo"),
        );
        assert_ref_status(
            no_repo_corroboration,
            "file",
            "src/lib.rs",
            "unresolved",
            Some("no_repo"),
        );
        assert_ref_status(
            no_repo_corroboration,
            "pr",
            "#7",
            "unresolved",
            Some("needs_network"),
        );
    }

    fn run_git_setup(cwd: &Path, args: &[&str]) -> String {
        let output = std::process::Command::new("git")
            .args(args)
            .current_dir(cwd)
            .env("GIT_TERMINAL_PROMPT", "0")
            .env("GIT_PAGER", "cat")
            .env("PAGER", "cat")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "git setup failed: {}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout).trim().to_string()
    }

    fn run_git(repo: &Path, args: &[&str]) -> String {
        let output = std::process::Command::new("git")
            .arg("-C")
            .arg(repo)
            .args(args)
            .env("GIT_TERMINAL_PROMPT", "0")
            .env("GIT_PAGER", "cat")
            .env("PAGER", "cat")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "git command failed: git -C {} {}\n{}\n{}",
            repo.display(),
            args.join(" "),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout).trim().to_string()
    }

    #[derive(Debug, PartialEq, Eq)]
    struct GitSnapshot {
        head: String,
        refs: String,
        index: String,
        status: String,
    }

    fn git_snapshot(repo: &Path) -> GitSnapshot {
        GitSnapshot {
            head: run_git(repo, &["rev-parse", "HEAD"]),
            refs: run_git(repo, &["for-each-ref", "--format=%(refname):%(objectname)"]),
            index: run_git(repo, &["ls-files", "-s"]),
            status: run_git(repo, &["status", "--porcelain=v1", "-z"]),
        }
    }

    fn assert_ref_status(
        corroboration: &Corroboration,
        kind: &str,
        reference: &str,
        status: &str,
        reason: Option<&str>,
    ) {
        let found = corroboration
            .refs
            .iter()
            .find(|candidate| candidate.kind == kind && candidate.reference == reference)
            .unwrap_or_else(|| panic!("missing corroborated ref {kind} {reference}"));
        assert_eq!(found.status, status);
        assert_eq!(found.reason.as_deref(), reason);
    }

    fn clear_git_invocations() {
        git_invocations().lock().unwrap().clear();
    }

    fn captured_git_invocations() -> Vec<Vec<String>> {
        git_invocations().lock().unwrap().clone()
    }

    fn assert_no_network_git_commands(commands: &[Vec<String>]) {
        assert!(
            !commands.is_empty(),
            "corroboration should have used local git read commands"
        );
        for command in commands {
            let Some(operation) = command.first().map(String::as_str) else {
                continue;
            };
            assert!(
                matches!(operation, "rev-parse" | "cat-file" | "log" | "ls-files"),
                "unexpected git operation in corroboration path: {command:?}"
            );
            assert!(
                !matches!(operation, "fetch" | "pull" | "ls-remote"),
                "network-capable git command must not run: {command:?}"
            );
        }
    }

    #[test]
    fn date_or_duration_filters_and_purge_before_use_normalized_thresholds() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();

        for (session_id, message_id, captured_at, prompt) in [
            (
                "old-session",
                "old-message",
                "2020-01-01T00:00:00Z",
                "datefilter old marker",
            ),
            (
                "new-session",
                "new-message",
                "2099-01-01T00:00:00Z",
                "datefilter new marker",
            ),
        ] {
            let event = envelope_from_backfill_payload(
                Tool::Claude,
                Path::new("/tmp/datefilter.jsonl"),
                0,
                json!({
                    "session_id": session_id,
                    "hook_event_name": "UserPromptSubmit",
                    "message_id": message_id,
                    "captured_at": captured_at,
                    "cwd": "/tmp/nabu-fixture",
                    "project_root": "/tmp/nabu-fixture",
                    "prompt": prompt
                }),
                &BackfillParseContext::default(),
            )
            .unwrap();
            append_prepared_event(&home, event).unwrap();
        }

        index_once(&home).unwrap();

        let recent = search_history_filtered(
            &home,
            "datefilter",
            SearchOptions {
                since: Some("1d".to_string()),
                limit: 10,
                ..SearchOptions::default()
            },
        )
        .unwrap();
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].session_id, "new-session");

        let sessions = list_sessions(&home, Some(Tool::Claude), None, Some("1d"), 10).unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].session_id, "new-session");

        let report = purge_before(&home, "2021-01-01").unwrap();
        assert_eq!(report.indexed_events_removed, 1);
        assert_eq!(
            search_history(&home, "datefilter old marker", 10)
                .unwrap()
                .len(),
            0
        );
        assert_eq!(
            search_history(&home, "datefilter new marker", 10)
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn search_filters_apply_session_type_file_and_command() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();

        ingest_hook_event(
            &home,
            Tool::Claude,
            json!({
                "session_id": "filter-session",
                "hook_event_name": "PreToolUse",
                "message_id": "command-filter-1",
                "cwd": "/tmp/nabu-fixture",
                "project_root": "/tmp/nabu-fixture",
                "tool_name": "bash",
                "command": "cargo test --workspace",
                "input": "command filter marker"
            }),
        )
        .unwrap();
        ingest_hook_event(
            &home,
            Tool::Opencode,
            json!({
                "session_id": "file-session",
                "event": "file.edited",
                "id": "file-filter-1",
                "cwd": "/tmp/nabu-fixture",
                "project_root": "/tmp/nabu-fixture",
                "path": "/tmp/nabu-fixture/src/auth.rs",
                "diff": "file filter marker"
            }),
        )
        .unwrap();
        index_once(&home).unwrap();

        let command_results = search_history_filtered(
            &home,
            "command filter marker",
            SearchOptions {
                session_id: Some("filter-session".to_string()),
                canonical_type: Some("tool.call".to_string()),
                command: Some("cargo test".to_string()),
                limit: 10,
                ..SearchOptions::default()
            },
        )
        .unwrap();
        assert_eq!(command_results.len(), 1);
        assert_eq!(command_results[0].canonical_type, "tool.call");

        let wrong_command = search_history_filtered(
            &home,
            "command filter marker",
            SearchOptions {
                command: Some("npm install".to_string()),
                limit: 10,
                ..SearchOptions::default()
            },
        )
        .unwrap();
        assert!(wrong_command.is_empty());

        let file_results = search_history_filtered(
            &home,
            "file filter marker",
            SearchOptions {
                file: Some("src/auth.rs".to_string()),
                canonical_type: Some("file.changed".to_string()),
                limit: 10,
                ..SearchOptions::default()
            },
        )
        .unwrap();
        assert_eq!(file_results.len(), 1);
        assert_eq!(file_results[0].session_id, "file-session");
    }

    #[test]
    fn search_defaults_are_citation_first_and_full_payload_is_opt_in() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();

        ingest_hook_event(
            &home,
            Tool::Claude,
            json!({
                "session_id": "citation-session",
                "hook_event_name": "UserPromptSubmit",
                "message_id": "citation-1",
                "cwd": "/tmp/nabu-fixture",
                "project_root": "/tmp/nabu-fixture",
                "prompt": format!("{} needle-centered citation marker {}", "prefix ".repeat(80), "suffix ".repeat(80))
            }),
        )
        .unwrap();
        index_once(&home).unwrap();
        let db_path = home.join("index").join("harness.db");
        let conn = open_index(&db_path).unwrap();
        let payload_json: Option<String> = conn
            .query_row("SELECT payload_json FROM events LIMIT 1", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert!(payload_json.is_none());

        let default_page = search_history_page(
            &home,
            "needle-centered citation marker",
            SearchOptions {
                max_snippet_chars: 48,
                ..SearchOptions::default()
            },
        )
        .unwrap();
        assert_eq!(default_page.returned, 1);
        assert_eq!(default_page.max_snippet_chars_applied, 48);
        assert!(default_page.results[0].payload.is_null());
        assert!(default_page.results[0].score > 0.0);
        assert!(default_page.results[0].snippet.contains("needle-centered"));
        assert!(default_page.results[0].snippet.chars().count() <= 48);

        let full_page = search_history_page(
            &home,
            "needle-centered citation marker",
            SearchOptions {
                include_payload: true,
                max_snippet_chars: 5_000,
                ..SearchOptions::default()
            },
        )
        .unwrap();
        assert_eq!(full_page.max_snippet_chars_applied, 1_000);
        assert!(full_page.results[0].payload.get("prompt").is_some());
    }

    #[test]
    fn payload_hydration_uses_raw_offset_and_falls_back_to_line_scan() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();

        for line in 1..=4 {
            ingest_hook_event(
                &home,
                Tool::Claude,
                json!({
                    "session_id": "offset-payload-session",
                    "hook_event_name": "UserPromptSubmit",
                    "message_id": format!("offset-payload-{line}"),
                    "cwd": "/tmp/nabu-fixture",
                    "project_root": "/tmp/nabu-fixture",
                    "prompt": format!("offset payload marker {line}")
                }),
            )
            .unwrap();
        }
        index_once(&home).unwrap();

        let raw_path = canonical_raw_path(&home, Tool::Claude, "offset-payload-session");
        let raw_file = raw_path.display().to_string();
        let offset = raw_offset_for_line(&raw_path, 3) as i64;
        let scanned = raw_envelope_for_line_scan(&raw_path, 4).unwrap();
        let sought = raw_envelope_for_pointer(&raw_file, 4, Some(offset)).unwrap();
        let fallback = raw_envelope_for_pointer(&raw_file, 4, Some(offset + 1)).unwrap();

        assert_eq!(sought, scanned);
        assert_eq!(fallback, scanned);
        assert_eq!(
            payload_for_raw_pointer(&raw_file, 4, Some(offset))
                .unwrap()
                .get("prompt"),
            Some(&json!("offset payload marker 4"))
        );
    }

    #[test]
    fn search_payload_hydration_uses_grouped_raw_offsets() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();

        for line in 1..=3 {
            ingest_hook_event(
                &home,
                Tool::Claude,
                json!({
                    "session_id": "grouped-payload-session",
                    "hook_event_name": "UserPromptSubmit",
                    "message_id": format!("grouped-payload-{line}"),
                    "cwd": "/tmp/nabu-fixture",
                    "project_root": "/tmp/nabu-fixture",
                    "prompt": format!("grouped payload shared marker {line}")
                }),
            )
            .unwrap();
        }
        index_once(&home).unwrap();

        let page = search_history_page(
            &home,
            "grouped payload shared marker",
            SearchOptions {
                include_payload: true,
                limit: 3,
                dedupe: false,
                ..SearchOptions::default()
            },
        )
        .unwrap();

        assert_eq!(page.returned, 3);
        let prompts = page
            .results
            .iter()
            .map(|result| {
                result
                    .payload
                    .get("prompt")
                    .and_then(Value::as_str)
                    .unwrap()
                    .to_string()
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(prompts.len(), 3);
        assert!(prompts.contains("grouped payload shared marker 1"));
        assert!(prompts.contains("grouped payload shared marker 2"));
        assert!(prompts.contains("grouped payload shared marker 3"));
    }

    #[test]
    fn search_auto_falls_back_to_lexical_and_forced_hybrid_errors_without_semantic_backend() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();

        ingest_hook_event(
            &home,
            Tool::Claude,
            json!({
                "session_id": "mode-session",
                "hook_event_name": "UserPromptSubmit",
                "message_id": "mode-1",
                "prompt": "search mode lexical fallback marker"
            }),
        )
        .unwrap();
        index_once(&home).unwrap();

        let auto_page = search_history_page(
            &home,
            "search mode lexical fallback marker",
            SearchOptions {
                mode: SearchMode::Auto,
                ..SearchOptions::default()
            },
        )
        .unwrap();
        assert_eq!(auto_page.mode_requested, SearchMode::Auto);
        assert_eq!(auto_page.mode_applied, SearchMode::Lexical);
        assert!(!auto_page.semantic_available);
        assert_eq!(auto_page.returned, 1);

        let lexical_page = search_history_page(
            &home,
            "search mode lexical fallback marker",
            SearchOptions {
                mode: SearchMode::Lexical,
                ..SearchOptions::default()
            },
        )
        .unwrap();
        assert_eq!(lexical_page.mode_applied, SearchMode::Lexical);
        assert_eq!(lexical_page.returned, 1);

        let error = search_history_page(
            &home,
            "search mode lexical fallback marker",
            SearchOptions {
                mode: SearchMode::Hybrid,
                ..SearchOptions::default()
            },
        )
        .unwrap_err();
        assert!(matches!(error, Error::SemanticUnavailable(_)));
    }

    #[test]
    fn embedding_units_are_structured_and_exclude_tool_output_noise() {
        let payload = json!({
            "tool_name": "shell",
            "command": "cargo test --workspace",
            "status": "failed",
            "stdout": "very long compiler output that should remain lexical-only",
            "stderr": "more noisy output that should not become a vector unit"
        });
        let document = search_document_for_event(CanonicalType::ToolResult, &payload);

        let units = embedding_units_for_document(&document);

        assert_eq!(units.len(), 1);
        assert_eq!(units[0].kind, EmbeddingUnitKind::ToolIntent);
        assert!(units[0].text.contains("cargo test --workspace"));
        assert!(!units[0].text.contains("compiler output"));
        assert_eq!(units[0].text_hash, sha256_hex(units[0].text.as_bytes()));
    }

    #[test]
    fn search_and_session_exclude_deltas_by_default_and_restore_on_opt_in() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();

        ingest_hook_event(
            &home,
            Tool::Claude,
            json!({
                "session_id": "delta-session",
                "hook_event_name": "MessageDisplay",
                "message_id": "delta-message",
                "index": 0,
                "final": false,
                "cwd": "/tmp/nabu-fixture",
                "project_root": "/tmp/nabu-fixture",
                "delta": "delta-only fixture marker"
            }),
        )
        .unwrap();
        ingest_hook_event(
            &home,
            Tool::Claude,
            json!({
                "session_id": "delta-session",
                "hook_event_name": "MessageDisplay",
                "message_id": "final-message",
                "index": 1,
                "final": true,
                "cwd": "/tmp/nabu-fixture",
                "project_root": "/tmp/nabu-fixture",
                "message": "final fixture marker"
            }),
        )
        .unwrap();
        index_once(&home).unwrap();

        assert!(search_history(&home, "delta-only fixture marker", 10)
            .unwrap()
            .is_empty());
        let delta_search = search_history_page(
            &home,
            "delta-only fixture marker",
            SearchOptions {
                include_deltas: true,
                ..SearchOptions::default()
            },
        )
        .unwrap();
        assert_eq!(delta_search.results[0].canonical_type, "assistant.delta");

        let default_session = get_session_page(
            &home,
            Tool::Claude,
            "delta-session",
            SessionOptions::default(),
        )
        .unwrap();
        assert!(default_session
            .events
            .iter()
            .all(|event| event.canonical_type != "assistant.delta"));

        let full_session = get_session_page(
            &home,
            Tool::Claude,
            "delta-session",
            SessionOptions {
                include_deltas: true,
                ..SessionOptions::default()
            },
        )
        .unwrap();
        assert_eq!(full_session.events[0].canonical_type, "assistant.delta");
        assert!(
            export_session_markdown_with_options(&home, Tool::Claude, "delta-session", false)
                .unwrap()
                .contains("delta-only fixture marker")
        );
    }

    #[test]
    fn session_context_window_clamps_and_wins_over_after_raw_line() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();

        for line in 1..=5 {
            ingest_hook_event(
                &home,
                Tool::Claude,
                json!({
                    "session_id": "window-session",
                    "hook_event_name": "UserPromptSubmit",
                    "message_id": format!("window-{line}"),
                    "cwd": "/tmp/nabu-fixture",
                    "project_root": "/tmp/nabu-fixture",
                    "prompt": format!("window marker line {line}")
                }),
            )
            .unwrap();
        }
        index_once(&home).unwrap();

        let window = get_session_page(
            &home,
            Tool::Claude,
            "window-session",
            SessionOptions {
                around_raw_line: Some(3),
                after_raw_line: Some(4),
                before: 1,
                after: 1,
                ..SessionOptions::default()
            },
        )
        .unwrap();
        assert_eq!(window.mode, "window");
        assert_eq!(
            window
                .events
                .iter()
                .map(|event| event.raw_line)
                .collect::<Vec<_>>(),
            vec![2, 3, 4]
        );

        let clamped = get_session_page(
            &home,
            Tool::Claude,
            "window-session",
            SessionOptions {
                around_raw_line: Some(1),
                before: 10,
                after: 0,
                ..SessionOptions::default()
            },
        )
        .unwrap();
        assert_eq!(clamped.events[0].raw_line, 1);
    }

    #[test]
    fn search_dedupes_twins_only_at_retrieval_layer() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        let source = temp.path().join("codex-sessions");
        init_home(&home).unwrap();
        fs::create_dir_all(&source).unwrap();

        let session_id = "019b0000-0000-7000-8000-000000000001";
        fs::write(
            source.join(format!("rollout-2026-06-18T00-00-00-{session_id}.jsonl")),
            format!(
                "{{\"timestamp\":\"2026-06-18T00:00:00Z\",\"type\":\"session_meta\",\"payload\":{{\"id\":\"{session_id}\",\"cwd\":\"/tmp/native-codex\"}}}}\n\
                 {{\"timestamp\":\"2026-06-18T00:00:01Z\",\"type\":\"response_item\",\"payload\":{{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{{\"type\":\"output_text\",\"text\":\"twinned codex answer marker\"}}]}}}}\n\
                 {{\"timestamp\":\"2026-06-18T00:00:01Z\",\"type\":\"event_msg\",\"payload\":{{\"type\":\"agent_message\",\"message\":\"twinned codex answer marker\"}}}}\n"
            ),
        )
        .unwrap();
        backfill_since(&home, Some(Tool::Codex), &source, None).unwrap();
        index_once(&home).unwrap();

        let deduped = search_history_page(
            &home,
            "twinned codex answer marker",
            SearchOptions::default(),
        )
        .unwrap();
        assert_eq!(deduped.results.len(), 1);
        assert_eq!(deduped.results[0].also_at.len(), 1);

        let not_deduped = search_history_page(
            &home,
            "twinned codex answer marker",
            SearchOptions {
                dedupe: false,
                ..SearchOptions::default()
            },
        )
        .unwrap();
        assert_eq!(not_deduped.results.len(), 2);

        let session =
            get_session_page(&home, Tool::Codex, session_id, SessionOptions::default()).unwrap();
        assert_eq!(
            session
                .events
                .iter()
                .filter(|event| event.text.contains("twinned codex answer marker"))
                .count(),
            2
        );
    }

    #[test]
    fn doctor_fast_and_deep_report_their_integrity_scope() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();
        ingest_hook_event(
            &home,
            Tool::Claude,
            json!({
                "session_id": "doctor-session",
                "hook_event_name": "UserPromptSubmit",
                "message_id": "doctor-1",
                "prompt": "doctor marker"
            }),
        )
        .unwrap();
        index_once(&home).unwrap();

        let fast = doctor_with_options(&home, false);
        assert_eq!(fast.level, "fast");
        assert_eq!(fast.integrity, "structural");
        assert!(fast.index.ok);
        assert!(fast.index.message.contains("core tables"));
        assert!(fast.stats.is_none());
        assert!(fast.latest_captured_events["claude"].is_some());

        let deep = doctor_with_options(&home, true);
        assert_eq!(deep.level, "deep");
        assert_eq!(deep.integrity, "full");
        assert!(deep.index.message.contains("integrity_check"));
        assert_eq!(deep.stats.unwrap().events, 1);

        let db_path = home.join("index").join("harness.db");
        let conn = Connection::open(&db_path).unwrap();
        let plan = conn
            .query_row(
                "EXPLAIN QUERY PLAN
                 SELECT tool, session_id, canonical_type, captured_at, searchable_text, raw_file, raw_line, raw_offset
                 FROM events
                 WHERE tool = 'claude'
                 ORDER BY captured_at DESC, id DESC
                 LIMIT 1",
                [],
                |row| row.get::<_, String>(3),
            )
            .unwrap();
        assert!(plan.contains("idx_events_tool_captured"), "{plan}");
    }

    #[test]
    fn set_opencode_server_url_round_trips_and_preserves_other_settings() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();
        let config = home.join("config.toml");

        // A config with an unrelated key and a comment that must survive edits.
        fs::write(
            &config,
            "schema_version = 1\n# keep me\n\n[opencode]\n# server_url = \"http://127.0.0.1:4096\"\n",
        )
        .unwrap();

        // Set activates the commented seed and is readable through the reader.
        // Read via the config parser directly so ambient env vars can't shadow it.
        set_opencode_server_url(&home, Some("http://localhost:9999")).unwrap();
        assert_eq!(
            read_opencode_server_url_from_config(&config)
                .unwrap()
                .as_deref(),
            Some("http://localhost:9999")
        );
        let after_set = fs::read_to_string(&config).unwrap();
        assert!(after_set.contains("schema_version = 1"));
        assert!(after_set.contains("# keep me"));
        assert!(after_set.contains("server_url = \"http://localhost:9999\""));

        // Idempotent: setting the same value does not rewrite the file.
        set_opencode_server_url(&home, Some("http://localhost:9999")).unwrap();
        assert_eq!(fs::read_to_string(&config).unwrap(), after_set);

        // Clear removes the active line but keeps the rest.
        set_opencode_server_url(&home, None).unwrap();
        assert_eq!(read_opencode_server_url_from_config(&config).unwrap(), None);
        let after_clear = fs::read_to_string(&config).unwrap();
        assert!(after_clear.contains("schema_version = 1"));
        assert!(after_clear.contains("# keep me"));
        assert!(!after_clear.contains("server_url = \"http://localhost:9999\""));
    }

    #[test]
    fn set_opencode_server_url_appends_section_when_absent() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        std::fs::create_dir_all(&home).unwrap();
        // No config.toml yet — writer must create it and append the section.
        set_opencode_server_url(&home, Some("http://127.0.0.1:4096")).unwrap();
        assert_eq!(
            read_opencode_server_url_from_config(&home.join("config.toml"))
                .unwrap()
                .as_deref(),
            Some("http://127.0.0.1:4096")
        );
    }

    #[test]
    fn search_treats_hyphenated_queries_as_plain_text() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();

        ingest_hook_event(
            &home,
            Tool::Claude,
            json!({
                "session_id": "hyphen-search-session",
                "hook_event_name": "UserPromptSubmit",
                "message_id": "hyphen-search-1",
                "cwd": "/tmp/nabu-fixture",
                "project_root": "/tmp/nabu-fixture",
                "prompt": "nabu project setup goals and tasks"
            }),
        )
        .unwrap();
        index_once(&home).unwrap();

        let results = search_history(&home, "nabu project setup goals and tasks", 10).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].session_id, "hyphen-search-session");
    }

    #[test]
    fn search_rejects_queries_without_searchable_text() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();

        let error = search_history(&home, "-- : ()", 10).unwrap_err();

        assert!(
            matches!(error, Error::Validation(message) if message == "query must contain searchable text")
        );
    }

    #[test]
    fn codex_exec_json_ingest_preserves_delta_order_and_usage_metadata() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        init_home(&home).unwrap();

        let report = ingest_file(
            &home,
            Tool::Codex,
            Source::ExecJson,
            &repo.join("fixtures/codex/exec-json.jsonl"),
        )
        .unwrap();

        assert_eq!(report.appended_events, 5);
        let raw_path = canonical_raw_path(&home, Tool::Codex, "codex-exec-stream-session");
        let envelopes = raw_envelopes(&raw_path);
        assert!(envelopes
            .iter()
            .all(|event| event.source == Source::ExecJson));
        let deltas = envelopes
            .iter()
            .filter(|event| event.canonical_type == CanonicalType::AssistantDelta)
            .collect::<Vec<_>>();
        assert_eq!(
            deltas
                .iter()
                .map(|event| event.sequence)
                .collect::<Vec<_>>(),
            vec![Some(0), Some(1)]
        );
        assert!(deltas[0]
            .payload
            .get("delta")
            .and_then(Value::as_str)
            .unwrap()
            .ends_with("delta one"));
        assert!(deltas[1]
            .payload
            .get("delta")
            .and_then(Value::as_str)
            .unwrap()
            .ends_with("delta two"));

        index_once(&home).unwrap();
        let usage_results = search_history_page(
            &home,
            "total_tokens 42",
            SearchOptions {
                tool: Some(Tool::Codex),
                limit: 10,
                ..SearchOptions::default()
            },
        )
        .unwrap();
        assert_eq!(usage_results.results.len(), 1);
        assert_eq!(usage_results.results[0].canonical_type, "session.ended");
        let export = export_session_jsonl_with_options(
            &home,
            Tool::Codex,
            "codex-exec-stream-session",
            false,
        )
        .unwrap();
        assert!(export.contains("\"total_tokens\":42"));
    }

    #[test]
    fn codex_app_server_ingest_preserves_jsonrpc_payloads_and_delta_order() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        init_home(&home).unwrap();

        let report = ingest_file(
            &home,
            Tool::Codex,
            Source::AppServer,
            &repo.join("fixtures/codex/app-server-notifications.jsonl"),
        )
        .unwrap();

        assert_eq!(report.appended_events, 6);
        let raw_path = canonical_raw_path(&home, Tool::Codex, "codex-app-server-session");
        let envelopes = raw_envelopes(&raw_path);
        assert!(envelopes
            .iter()
            .all(|event| event.source == Source::AppServer));
        assert_eq!(envelopes[0].source_event_type, "thread/started");
        assert!(envelopes[0].payload.get("jsonrpc").is_some());
        assert!(envelopes
            .iter()
            .any(|event| event.canonical_type == CanonicalType::ToolCall));
        let deltas = envelopes
            .iter()
            .filter(|event| event.canonical_type == CanonicalType::AssistantDelta)
            .collect::<Vec<_>>();
        assert_eq!(
            deltas
                .iter()
                .map(|event| event.sequence)
                .collect::<Vec<_>>(),
            vec![Some(0), Some(1)]
        );
        assert!(deltas.iter().all(|event| event
            .source_event_id
            .as_deref()
            .unwrap()
            .contains(":delta")));
    }

    #[test]
    fn codex_streaming_and_hook_identity_dedupe_same_event_but_keep_deltas() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        let stream = temp.path().join("codex-stream.jsonl");
        let session_id = "codex-stream-identity-session";
        init_home(&home).unwrap();

        ingest_hook_event(
            &home,
            Tool::Codex,
            json!({
                "session_id": session_id,
                "type": "response_item",
                "payload": {
                    "id": "codex-stream-shared-item",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "codex stream shared identity marker"}]
                }
            }),
        )
        .unwrap();
        fs::write(
            &stream,
            format!(
                "{}\n{}\n",
                serde_json::to_string(&json!({
                    "timestamp": "2026-06-18T10:00:00Z",
                    "type": "item.completed",
                    "thread_id": session_id,
                    "item": {
                        "id": "codex-stream-shared-item",
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "codex stream shared identity marker"}]
                    }
                }))
                .unwrap(),
                serde_json::to_string(&json!({
                    "timestamp": "2026-06-18T10:00:01Z",
                    "type": "item/agentMessage/delta",
                    "thread_id": session_id,
                    "turn_id": "codex-stream-turn",
                    "message_id": "codex-stream-delta-message",
                    "sequence": 0,
                    "delta": "codex stream granularity marker"
                }))
                .unwrap()
            ),
        )
        .unwrap();

        let report = ingest_file(&home, Tool::Codex, Source::ExecJson, &stream).unwrap();

        assert_eq!(report.appended_events, 1);
        let envelopes = raw_envelopes(&canonical_raw_path(&home, Tool::Codex, session_id));
        assert_eq!(
            envelopes
                .iter()
                .filter(|event| event.source_event_id.as_deref()
                    == Some("codex-stream-shared-item"))
                .count(),
            1
        );
        assert!(envelopes
            .iter()
            .any(|event| event.canonical_type == CanonicalType::AssistantDelta));
    }

    #[test]
    fn opencode_server_messages_reconcile_gaps_without_spool_copy() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        let session_id = "opencode-server-reconcile-session";
        init_home(&home).unwrap();
        ingest_hook_event(
            &home,
            Tool::Opencode,
            json!({
                "session_id": session_id,
                "hook_event_name": "message.updated",
                "id": "opencode-server-shared-message",
                "role": "assistant",
                "text": "opencode server shared marker"
            }),
        )
        .unwrap();

        let report = ingest_opencode_server_messages(
            &home,
            session_id,
            json!([
                {
                    "id": "opencode-server-shared-message",
                    "sessionID": session_id,
                    "role": "assistant",
                    "text": "opencode server shared marker"
                },
                {
                    "id": "opencode-server-gap-message",
                    "sessionID": session_id,
                    "role": "assistant",
                    "worktree": "/Users/example/opencode-server-worktree",
                    "parts": [
                        {
                            "id": "opencode-server-gap-part",
                            "type": "text",
                            "text": "opencode server recovered part marker"
                        }
                    ]
                }
            ]),
        )
        .unwrap();

        assert_eq!(report.appended_events, 1);
        assert!(!home.join("spool/opencode-api").exists());
        let envelopes = raw_envelopes(&canonical_raw_path(&home, Tool::Opencode, session_id));
        assert_eq!(
            envelopes
                .iter()
                .filter(|event| event.source_event_id.as_deref()
                    == Some("opencode-server-shared-message"))
                .count(),
            1
        );
        let gap = envelopes
            .iter()
            .find(|event| {
                event.source_event_type == "message.part.updated"
                    && event.payload.pointer("/part/text").and_then(Value::as_str)
                        == Some("opencode server recovered part marker")
            })
            .unwrap();
        assert_eq!(
            gap.project_root.as_deref(),
            Some("/Users/example/opencode-server-worktree")
        );
        assert_eq!(
            gap.cwd.as_deref(),
            Some("/Users/example/opencode-server-worktree")
        );
        assert!(envelopes.iter().any(|event| {
            event.source_event_type == "message.part.updated"
                && event.payload.pointer("/part/text").and_then(Value::as_str)
                    == Some("opencode server recovered part marker")
        }));
    }

    fn raw_line_count(path: &Path) -> usize {
        fs::read_to_string(path).unwrap().lines().count()
    }

    fn dedupe_sidecar_entry_count(sidecar: &DedupeSidecarFiles) -> usize {
        fs::read_dir(&sidecar.buckets_dir)
            .unwrap()
            .map(|entry| raw_line_count(&entry.unwrap().path()))
            .sum()
    }

    fn raw_envelopes(path: &Path) -> Vec<EventEnvelope> {
        fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    fn raw_offset_for_line(path: &Path, zero_based_line: usize) -> u64 {
        let content = fs::read_to_string(path).unwrap();
        content
            .lines()
            .take(zero_based_line)
            .map(|line| line.len() as u64 + 1)
            .sum()
    }

    fn checkpoint_row_count(home: &Path) -> i64 {
        let db_path = home.join("index").join("harness.db");
        let conn = Connection::open(db_path).unwrap();
        conn.query_row("SELECT COUNT(*) FROM checkpoints", [], |row| row.get(0))
            .unwrap()
    }

    fn checkpoint_sidecar_count(home: &Path) -> usize {
        let dir = home.join("checkpoints");
        fs::read_dir(dir)
            .unwrap()
            .filter(|entry| {
                entry
                    .as_ref()
                    .ok()
                    .map(|entry| entry.path().is_file())
                    .unwrap_or(false)
            })
            .count()
    }

    fn discontinuity_count(home: &Path, tool: Tool, session_id: &str, reason: &str) -> usize {
        raw_envelopes(&canonical_raw_path(home, tool, session_id))
            .into_iter()
            .filter(|event| {
                event.canonical_type == CanonicalType::SourceDiscontinuity
                    && event.payload.get("reason").and_then(Value::as_str) == Some(reason)
            })
            .count()
    }

    fn claude_user_line(session_id: &str, uuid: &str, text: &str) -> String {
        serde_json::to_string(&json!({
            "type": "user",
            "sessionId": session_id,
            "uuid": uuid,
            "timestamp": "2026-06-18T12:00:00.000Z",
            "cwd": "/tmp/native-claude",
            "message": {
                "role": "user",
                "content": text
            }
        }))
        .unwrap()
            + "\n"
    }

    fn assert_reconciles_claude(home: &Path, temp_root: &Path) {
        let source = temp_root.join("reconcile-claude");
        fs::create_dir_all(&source).unwrap();
        let session_id = "aaaaaaa1-aaaa-4aaa-8aaa-aaaaaaaaaaa1";
        ingest_hook_event(
            home,
            Tool::Claude,
            json!({
                "session_id": session_id,
                "hook_event_name": "UserPromptSubmit",
                "event_id": "claude-reconcile-shared",
                "prompt": "claude reconcile shared marker"
            }),
        )
        .unwrap();
        ingest_hook_event(
            home,
            Tool::Claude,
            json!({
                "session_id": session_id,
                "hook_event_name": "MessageDisplay",
                "message_id": "claude-reconcile-delta",
                "index": 0,
                "final": false,
                "delta": "claude reconcile granularity marker"
            }),
        )
        .unwrap();
        fs::write(
            source.join(format!("{session_id}.jsonl")),
            format!(
                "{}{}{}",
                claude_user_line(
                    session_id,
                    "claude-reconcile-shared",
                    "claude reconcile shared marker"
                ),
                serde_json::to_string(&json!({
                    "type": "assistant",
                    "sessionId": session_id,
                    "timestamp": "2026-06-18T12:00:01.000Z",
                    "message": {
                        "id": "claude-reconcile-gap",
                        "role": "assistant",
                        "content": [{"type": "text", "text": "claude reconcile gap marker"}]
                    }
                }))
                .unwrap()
                    + "\n",
                serde_json::to_string(&json!({
                    "type": "assistant",
                    "sessionId": session_id,
                    "timestamp": "2026-06-18T12:00:02.000Z",
                    "message": {
                        "id": "claude-reconcile-final",
                        "role": "assistant",
                        "content": [{"type": "text", "text": "claude reconcile granularity marker"}]
                    }
                }))
                .unwrap()
                    + "\n"
            ),
        )
        .unwrap();

        let report = backfill_since(home, Some(Tool::Claude), &source, None).unwrap();
        assert_eq!(report.appended_events, 2);
        let envelopes = raw_envelopes(&canonical_raw_path(home, Tool::Claude, session_id));
        assert_eq!(
            envelopes
                .iter()
                .filter(|event| event.source_event_id.as_deref() == Some("claude-reconcile-shared"))
                .count(),
            1
        );
        assert!(envelopes.iter().any(|event| {
            event.canonical_type == CanonicalType::AssistantMessage
                && event.source_event_id.as_deref() == Some("claude-reconcile-gap")
        }));
        assert!(envelopes
            .iter()
            .any(|event| event.canonical_type == CanonicalType::AssistantDelta));
        assert!(envelopes.iter().any(|event| event.canonical_type
            == CanonicalType::AssistantMessage
            && event.source_event_id.as_deref() == Some("claude-reconcile-final")));
    }

    fn assert_reconciles_codex(home: &Path, temp_root: &Path) {
        let source = temp_root.join("reconcile-codex");
        fs::create_dir_all(&source).unwrap();
        let session_id = "bbbbbbb2-bbbb-4bbb-8bbb-bbbbbbbbbbb2";
        ingest_hook_event(
            home,
            Tool::Codex,
            json!({
                "session_id": session_id,
                "type": "response_item",
                "payload": {
                    "id": "codex-reconcile-shared",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "codex reconcile shared marker"}]
                }
            }),
        )
        .unwrap();
        fs::write(
            source.join(format!("rollout-2026-06-18T12-00-00-{session_id}.jsonl")),
            "{\"timestamp\":\"2026-06-18T12:00:00.000Z\",\"type\":\"response_item\",\"payload\":{\"id\":\"codex-reconcile-shared\",\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"codex reconcile shared marker\"}]}}\n\
                 {\"timestamp\":\"2026-06-18T12:00:01.000Z\",\"type\":\"response_item\",\"payload\":{\"id\":\"codex-reconcile-gap\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"codex reconcile gap marker\"}]}}\n",
        )
        .unwrap();

        let report = backfill_since(home, Some(Tool::Codex), &source, None).unwrap();
        assert_eq!(report.appended_events, 1);
        let envelopes = raw_envelopes(&canonical_raw_path(home, Tool::Codex, session_id));
        assert_eq!(
            envelopes
                .iter()
                .filter(|event| event.source_event_id.as_deref() == Some("codex-reconcile-shared"))
                .count(),
            1
        );
        assert!(envelopes
            .iter()
            .any(|event| event.source_event_id.as_deref() == Some("codex-reconcile-gap")));
    }

    fn assert_reconciles_opencode(home: &Path, temp_root: &Path) {
        let root = temp_root.join("reconcile-opencode");
        let session_id = "ccccccc3-cccc-4ccc-8ccc-ccccccccccc3";
        let message_dir = root.join("storage/message").join(session_id);
        fs::create_dir_all(&message_dir).unwrap();
        ingest_hook_event(
            home,
            Tool::Opencode,
            json!({
                "session_id": session_id,
                "event": "message.updated",
                "id": "opencode-reconcile-shared",
                "text": "opencode reconcile shared marker"
            }),
        )
        .unwrap();
        fs::write(
            message_dir.join("opencode-reconcile-shared.json"),
            serde_json::to_string_pretty(&json!({
                "id": "opencode-reconcile-shared",
                "sessionID": session_id,
                "role": "assistant",
                "text": "opencode reconcile shared marker"
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(
            message_dir.join("opencode-reconcile-gap.json"),
            serde_json::to_string_pretty(&json!({
                "id": "opencode-reconcile-gap",
                "sessionID": session_id,
                "role": "assistant",
                "text": "opencode reconcile gap marker"
            }))
            .unwrap(),
        )
        .unwrap();

        let report = backfill_since(home, Some(Tool::Opencode), &root, None).unwrap();
        assert_eq!(report.appended_events, 1);
        let envelopes = raw_envelopes(&canonical_raw_path(home, Tool::Opencode, session_id));
        assert_eq!(
            envelopes
                .iter()
                .filter(
                    |event| event.source_event_id.as_deref() == Some("opencode-reconcile-shared")
                )
                .count(),
            1
        );
        assert!(envelopes
            .iter()
            .any(|event| event.source_event_id.as_deref() == Some("opencode-reconcile-gap")));
    }

    fn valid_envelope_json() -> Value {
        json!({
            "schema_version": 1,
            "captured_at": "2026-06-17T12:00:00Z",
            "tool": "codex",
            "tool_version": null,
            "session_id": "session/one",
            "filename_session_id": "session_one",
            "turn_id": null,
            "message_id": null,
            "project_root": null,
            "cwd": "/tmp/nabu-fixture",
            "source": "hook",
            "source_event_type": "UserPromptSubmit",
            "canonical_type": "user.message",
            "source_event_id": null,
            "dedupe_key": "sha256:abc",
            "sequence": null,
            "raw_file": null,
            "raw_offset": null,
            "payload": {}
        })
    }
}
