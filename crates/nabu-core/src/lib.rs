//! Core history-keeping engine for the `nabu` CLI: ingest, indexing, search,
//! export, and maintenance over local coding-agent transcripts.
//!
//! This crate is published only so the `nabu` binary (the `nabu-cli` crate)
//! resolves its dependencies. It is not a stable public API — items may change
//! or be removed in any release with no semver guarantee. Depend on the `nabu`
//! CLI, not on this crate directly.
#![doc(hidden)]

use fs2::FileExt;
use rayon::prelude::*;
use rusqlite::OptionalExtension;
#[cfg(feature = "semantic")]
use rusqlite::{params_from_iter, types::Value as SqlValue};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{self, File, OpenOptions};
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
pub(crate) const SQLITE_SCHEMA: &str = include_str!("../schema.sql");
pub(crate) const MAX_INLINE_ENVELOPE_BYTES: usize = 16 * 1024 * 1024;
mod db;
pub(crate) use db::{
    ensure_semantic_vector_schema, initialize_database, open_index, table_count, table_exists,
};
const MAX_SEARCH_LIMIT: usize = 50;
/// Hard upper bound on per-result snippet length. Callers may request up to this
/// many characters; requests above it are clamped.
pub const MAX_SEARCH_SNIPPET_CHARS: usize = 1000;
/// Default per-result snippet length applied when a caller omits
/// `max_snippet_chars`. Sized for triage: ~500 chars (a few sentences of
/// match-centered context) is enough to tell a real bug from discussion of one
/// without a `get_session` round-trip, while staying far inside the MCP
/// response-size budget even at the maximum result count.
pub const DEFAULT_SEARCH_SNIPPET_CHARS: usize = 500;
const MAX_SESSION_LIMIT: usize = 500;
const MAX_CONTEXT_EVENTS_PER_SIDE: usize = 500;
const MAX_DIRECTORY_SIZE_DEPTH: usize = 64;
mod concept_expansion;
pub(crate) use concept_expansion::expand_query_terms;
mod semantic;
#[cfg(all(test, feature = "semantic"))]
pub(crate) use semantic::{
    bucket_unembedded_units, collect_unembedded_units, embed_unembedded_units_with_config,
    embedding_index_progress, estimated_embedding_token_count, vector_to_blob,
    EmbeddingWriteConfig, UnembeddedUnit,
};
#[cfg(test)]
pub(crate) use semantic::{
    document_embedding_input, model_file_expected_sha256, query_embedding_input,
    semantic_model_cache_path, sha256_hex_of_file, verify_file_sha256, SEMANTIC_MODEL_FILE_SHA256,
    SEMANTIC_MODEL_ID, SEMANTIC_MODEL_REMOTE_FILES, SEMANTIC_MODEL_REPO, SEMANTIC_MODEL_REVISION,
};
pub use semantic::{
    download_embedding_model_with_progress, embedding_model_disclosure, embedding_model_status,
    prune_embedding_cache,
};
pub(crate) use semantic::{
    embed_index_if_available_with_progress, insert_vector_unit_rows, semantic_search_available,
    vector_search_results, SEMANTIC_VECTOR_DIMENSIONS,
};

mod error;
pub use error::{Error, Result};

mod event;
pub use event::{
    summary_kind_for_canonical_str, CanonicalType, DedupeParts, EventEnvelope, Source, SummaryKind,
    Tool,
};

mod identity;
pub use identity::{dedupe_key, sanitize_session_id};
pub(crate) use identity::{hash_line, sha256_hex};

mod paths;
pub use paths::{canonical_raw_path, resolve_home};
pub(crate) use paths::{
    chmod, create_dir_0700, harness_home_for_raw_file, lock_path_for_raw_file, set_if_exists,
};

mod config;
pub(crate) use config::create_config_if_missing;
pub use config::{opencode_server_url, set_opencode_server_url};

mod semantic_api;
pub use semantic_api::{Embedder, EmbeddingUnit, EmbeddingUnitKind};

mod options;
pub(crate) use options::RankedSearchResult;
pub use options::{
    native_jsonl_line_command, AppendReport, BackfillCoverageSession, BackfillDryRunReport,
    BackfillImportPreview, BackfillProgress, BackfillReport, CorroboratedRef, Corroboration,
    CoverageSummary, DoctorCheck, DoctorReport, DoctorStats, EmbeddingDownloadProgress,
    EmbeddingDownloadReport, EmbeddingIndexProgress, EmbeddingModelDisclosure,
    EmbeddingModelStatus, EventOptions, EventPointer, FileIngestReport, FileTouch, IndexFreshness,
    IndexOptions, IndexReport, InitReport, PurgeAction, PurgeAllArtifact, PurgeAllOptions,
    PurgeAllReport, PurgeReport, PurgeTier, SearchContinuation, SearchMode, SearchOptions,
    SearchPage, SearchResult, SessionOptions, SessionPage, SessionSummary, StorageFootprint,
    StoredEvent, ToolUsage, SESSION_PROMPT_SNIPPET_CHARS, SESSION_TOP_FILES, SESSION_TOP_TOOLS,
};

mod purge;
pub use purge::{purge_all, purge_before, purge_session};

mod doctor;
pub(crate) use doctor::{directory_size, storage_footprint};
pub use doctor::{doctor, doctor_with_options, doctor_with_progress, index_freshness, DoctorStage};
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
    raw_index_checkpoint_is_current, raw_index_checkpoint_offset, source_file_metadata,
    write_raw_index_checkpoint,
};
#[cfg(test)]
pub(crate) use backfill::{backfill_dry_run, backfill_since};
pub use backfill::{backfill_dry_run_with_progress, backfill_since_with_progress};
mod ingest;
pub(crate) use ingest::{
    append_envelope_locked, append_envelopes_locked, load_full_dedupe_sidecar_events,
    read_raw_dedupe_snapshot, remove_dedupe_sidecar_for_raw_file, resolved_payload_for_envelope,
    sequence_for_payload, source_event_id_for_payload, DedupeSidecarFiles, ExistingRawEvent,
};
pub use ingest::{ingest_file, ingest_hook_event, ingest_opencode_server_messages, init_home};

mod index;
pub use index::{
    index_once, index_once_single_flight, index_once_with_options,
    index_once_with_options_and_progress, SingleFlightOutcome,
};
pub(crate) use index::{recalculate_all_session_counts, RawIndexFileReport};

mod search;
#[cfg(test)]
pub(crate) use search::corroborate::{extract_corroboration_candidates, git_invocations};
pub(crate) use search::corroborate_text;
#[cfg(test)]
pub(crate) use search::search_history_filtered;
#[cfg(feature = "semantic")]
pub(crate) use search::{
    match_centered_snippet, normalize_ref_filter, resolve_session_filter_ids,
    retrieval_key_for_text, unique_ranked_results_by_event,
};
pub use search::{search_history, search_history_page};

mod read;
pub(crate) use read::session_events;
pub use read::{get_event_by_pointer_with_options, get_session_page, latest_event, list_sessions};

mod export;
pub use export::{export_session_jsonl_with_options, export_session_markdown_with_options};

mod redact;
pub use redact::{redact_export_json, redact_export_text};
pub(crate) use redact::{redact_json_value, redact_text};

mod provenance;
pub(crate) use provenance::extract_refs;

mod raw;
pub(crate) use raw::{
    open_raw_offset_reader, payload_for_raw_pointer, raw_envelope_for_line_scan,
    raw_envelope_for_pointer, read_raw_envelope_at_offset, session_raw_file,
};

mod document;
pub(crate) use document::{
    canonical_type_for_payload, compaction_state_for, file_paths_for_payload, hook_event_name,
    identity_payload, message_text_for_document, normalize_identity_text, role_for,
    search_document_for_event, string_field, tool_status_for, SearchDocument,
};
// Used only by the cfg(semantic) vector pipeline and a default-build unit test.
#[cfg(any(feature = "semantic", test))]
pub(crate) use document::embedding_units_for_document;

#[cfg(test)]
mod tests;
