//! Crate error type and `Result` alias.

use std::path::PathBuf;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("validation error: {0}")]
    Validation(String),
    /// A resource addressed by an exact key was absent. Carries the resource
    /// kind and the key so downstream crates classify not-found by matching the
    /// variant rather than by sniffing message text.
    #[error("{0}")]
    NotFound(NotFound),
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

/// What was looked up, and by which key, when a not-found lookup failed. Each
/// variant's `Display` reproduces the exact human-facing message the lookup
/// sites previously built inline, so CLI/text output is byte-for-byte unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NotFound {
    /// No `sessions` row for `tool`/`session_id`.
    Session { tool: String, session_id: String },
    /// No indexed event for `tool`/`session_id`.
    Event { tool: String, session_id: String },
    /// No raw JSONL line `line` in the capture file at `path`.
    RawLine { line: i64, path: PathBuf },
}

impl std::fmt::Display for NotFound {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NotFound::Session { tool, session_id } => {
                write!(formatter, "session not found for {tool}:{session_id}")
            }
            NotFound::Event { tool, session_id } => {
                write!(formatter, "event not found for {tool}:{session_id}")
            }
            NotFound::RawLine { line, path } => {
                write!(formatter, "raw line {line} not found in {}", path.display())
            }
        }
    }
}
