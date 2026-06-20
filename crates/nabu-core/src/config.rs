//! OpenCode `config.toml` I/O: reading and atomically rewriting the
//! `[opencode] server_url` setting while preserving every other byte.

use crate::{chmod, Error, Result};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;

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

pub(crate) fn create_config_if_missing(path: &Path) -> Result<()> {
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

pub(crate) fn read_opencode_server_url_from_config(path: &Path) -> Result<Option<String>> {
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
    let validated_url = url.map(validate_opencode_server_url).transpose()?;
    let updated = rewrite_opencode_server_url(&content, validated_url);
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
    let new_line = url.map(|value| format!("server_url = {}", toml_basic_string(value)));
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

fn validate_opencode_server_url(url: &str) -> Result<&str> {
    let value = url.trim();
    if value.is_empty() {
        return Err(Error::Validation(
            "OpenCode server URL must not be empty".to_string(),
        ));
    }
    if value != url {
        return Err(Error::Validation(
            "OpenCode server URL must not include leading or trailing whitespace".to_string(),
        ));
    }
    if value
        .chars()
        .any(|character| character.is_control() || character.is_whitespace())
    {
        return Err(Error::Validation(
            "OpenCode server URL must not contain whitespace or control characters".to_string(),
        ));
    }
    if value.contains('"') || value.contains('\\') {
        return Err(Error::Validation(
            "OpenCode server URL must not contain quotes or backslashes".to_string(),
        ));
    }
    let Some(rest) = value.strip_prefix("http://") else {
        return Err(Error::Validation(
            "OpenCode server URL must use http:// for local reconciliation".to_string(),
        ));
    };
    let authority = rest.split('/').next().unwrap_or("");
    if authority.is_empty() {
        return Err(Error::Validation(
            "OpenCode server URL host must not be empty".to_string(),
        ));
    }
    if let Some((host, port)) = authority.rsplit_once(':') {
        if host.is_empty() || port.is_empty() || port.parse::<u16>().is_err() {
            return Err(Error::Validation(
                "OpenCode server URL port must be a valid TCP port".to_string(),
            ));
        }
    }
    Ok(value)
}

fn toml_basic_string(value: &str) -> String {
    let mut output = String::with_capacity(value.len() + 2);
    output.push('"');
    for character in value.chars() {
        match character {
            '\u{08}' => output.push_str("\\b"),
            '\t' => output.push_str("\\t"),
            '\n' => output.push_str("\\n"),
            '\u{0c}' => output.push_str("\\f"),
            '\r' => output.push_str("\\r"),
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            character if character.is_control() => {
                output.push_str(&format!("\\u{:04X}", character as u32));
            }
            character => output.push(character),
        }
    }
    output.push('"');
    output
}
