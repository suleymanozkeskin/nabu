//! Session export to JSONL and Markdown.

use crate::{
    canonical_raw_path, redact_json_value, redact_text, session_events, Error, Result, Tool,
};
use serde_json::Value;
use std::fs::File;
use std::io::Read;
use std::path::Path;

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
    if !redact {
        return Ok(content);
    }
    // Redact per line, key-aware: parse each JSONL line as JSON and run
    // redact_json_value, which both masks values under sensitive keys (e.g.
    // "api_key") and applies the text-pattern rules to every string leaf.
    // Lines that fail to parse (malformed or non-JSON) fall back to the raw
    // text-pattern redaction so no line is ever left unredacted.
    let redacted_lines: Vec<String> = content
        .lines()
        .map(|line| {
            if line.trim().is_empty() {
                return line.to_string();
            }
            match serde_json::from_str::<Value>(line) {
                Ok(value) => serde_json::to_string(&redact_json_value(value))
                    .unwrap_or_else(|_| redact_text(line)),
                Err(_) => redact_text(line),
            }
        })
        .collect();
    let mut output = redacted_lines.join("\n");
    if content.ends_with('\n') {
        output.push('\n');
    }
    Ok(output)
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
