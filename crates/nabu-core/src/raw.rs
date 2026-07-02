//! Raw JSONL capture-file readers: locate a session's raw file and parse
//! stored event envelopes by pointer, line, or byte offset.

use crate::{
    open_index, resolved_payload_for_envelope, Error, EventEnvelope, NotFound, Result, Tool,
};
use rusqlite::OptionalExtension;
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::{Path, PathBuf};

pub(crate) fn session_raw_file(home: &Path, tool: Tool, session_id: &str) -> Result<String> {
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
        Error::NotFound(NotFound::Session {
            tool: tool.as_str().to_string(),
            session_id: session_id.to_string(),
        })
    })
}

pub(crate) fn read_raw_line(path: &Path, requested_line: i64) -> Result<String> {
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
    Err(Error::NotFound(NotFound::RawLine {
        line: requested_line,
        path: path.to_path_buf(),
    }))
}

pub(crate) fn raw_envelope_for_pointer(
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

pub(crate) fn raw_envelope_for_line_scan(raw_path: &Path, raw_line: i64) -> Result<EventEnvelope> {
    let raw_text = read_raw_line(raw_path, raw_line)?;
    Ok(serde_json::from_str(raw_text.trim_end())?)
}

pub(crate) fn open_raw_offset_reader(raw_path: &Path) -> Result<BufReader<File>> {
    let file = File::open(raw_path).map_err(|source| Error::Io {
        path: raw_path.to_path_buf(),
        source,
    })?;
    Ok(BufReader::new(file))
}

pub(crate) fn read_raw_envelope_at_offset(
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

pub(crate) fn raw_envelope_matches_pointer(
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

pub(crate) fn payload_for_raw_pointer(
    raw_file: &str,
    raw_line: i64,
    raw_offset: Option<i64>,
) -> Result<Value> {
    let raw_path = PathBuf::from(raw_file);
    let envelope = raw_envelope_for_pointer(raw_file, raw_line, raw_offset)?;
    resolved_payload_for_envelope(&raw_path, &envelope)
}
