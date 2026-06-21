//! Session export to JSONL and Markdown.

use crate::{canonical_raw_path, redact_text, session_events, Error, Result, Tool};
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
    if redact {
        Ok(redact_text(&content))
    } else {
        Ok(content)
    }
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
