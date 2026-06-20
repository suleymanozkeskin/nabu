//! Event envelope, dedupe parts, and the canonical event enums.

use crate::{sanitize_session_id, Error, Result, SCHEMA_VERSION};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt;
use std::str::FromStr;

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
