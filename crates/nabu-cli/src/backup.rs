//! Atomic config writes, timestamped backups, and POSIX-mode helpers.
//!
//! Before mutating a tool's CLI config, callers snapshot it here: `backup_cli_
//! config` copies the file to a `*.nabu-backup.<stamp>.<sha8>.bak` sibling and
//! appends a manifest record under `<home>/backups/manifest.jsonl`. Writes go
//! through `write_text_config`, which creates parents 0700 and chmods the file
//! to the requested mode. `text_diff` renders the before/after shown in CLI
//! output. The `chmod_path`/`file_mode_or` pair is a no-op on non-unix.

use nabu_core::{Error, Tool};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

pub(crate) fn read_text_or_empty(path: &PathBuf) -> nabu_core::Result<String> {
    if !path.exists() {
        return Ok(String::new());
    }
    fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.clone(),
        source,
    })
}

pub(crate) fn write_text_config(path: &PathBuf, content: &str, mode: u32) -> nabu_core::Result<()> {
    let final_mode = file_mode_or(path, mode)?;
    if let Some(parent) = path.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent).map_err(|source| Error::Io {
                path: parent.to_path_buf(),
                source,
            })?;
            chmod_path(parent, 0o700)?;
        }
    }
    // Atomic replace: write a sibling temp file in the target's directory, set
    // its mode, then rename over the target. A same-directory rename is atomic
    // on the destination filesystem, so a reader never observes a half-written
    // config even if the process is interrupted mid-write.
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("config");
    let tmp_path = parent.join(format!(".{file_name}.nabu-tmp.{}", std::process::id()));
    fs::write(&tmp_path, content).map_err(|source| Error::Io {
        path: tmp_path.clone(),
        source,
    })?;
    chmod_path(&tmp_path, final_mode)?;
    if let Err(source) = fs::rename(&tmp_path, path) {
        let _ = fs::remove_file(&tmp_path);
        return Err(Error::Io {
            path: path.clone(),
            source,
        });
    }
    Ok(())
}

pub(crate) fn backup_cli_config(
    home: &Path,
    tool: Tool,
    operation: &str,
    path: &Path,
) -> nabu_core::Result<()> {
    if !path.exists() {
        return Ok(());
    }
    let content = fs::read(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let now = OffsetDateTime::now_utc();
    let created_at = now.format(&Rfc3339)?;
    let stamp = backup_stamp(now);
    let hash = sha256_hex(&content);
    let backup_path = path.with_file_name(format!(
        "{}.nabu-backup.{}.{}.bak",
        path.file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("config"),
        stamp,
        &hash[..8]
    ));
    fs::write(&backup_path, &content).map_err(|source| Error::Io {
        path: backup_path.clone(),
        source,
    })?;
    chmod_path(&backup_path, 0o600)?;

    let backups_dir = home.join("backups");
    fs::create_dir_all(&backups_dir).map_err(|source| Error::Io {
        path: backups_dir.clone(),
        source,
    })?;
    chmod_path(&backups_dir, 0o700)?;
    let manifest_path = backups_dir.join("manifest.jsonl");
    let record = json!({
        "created_at": created_at,
        "tool": tool.as_str(),
        "operation": operation,
        "original_path": path.display().to_string(),
        "backup_path": backup_path.display().to_string(),
        "sha256": hash
    });
    let mut manifest = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&manifest_path)
        .map_err(|source| Error::Io {
            path: manifest_path.clone(),
            source,
        })?;
    manifest
        .write_all(serde_json::to_string(&record)?.as_bytes())
        .map_err(|source| Error::Io {
            path: manifest_path.clone(),
            source,
        })?;
    manifest.write_all(b"\n").map_err(|source| Error::Io {
        path: manifest_path.clone(),
        source,
    })?;
    chmod_path(&manifest_path, 0o600)
}

pub(crate) fn text_diff(before: &str, after: &str) -> String {
    let mut diff = String::with_capacity(before.len() + after.len() + 24);
    diff.push_str("--- before\n");
    diff.push_str(before);
    diff.push_str("\n--- after\n");
    diff.push_str(after);
    diff.push('\n');
    diff
}

fn backup_stamp(time: OffsetDateTime) -> String {
    format!(
        "{:04}{:02}{:02}T{:02}{:02}{:02}Z",
        time.year(),
        time.month() as u8,
        time.day(),
        time.hour(),
        time.minute(),
        time.second()
    )
}

fn sha256_hex(content: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content);
    hex::encode(hasher.finalize())
}

#[cfg(unix)]
fn file_mode_or(path: &Path, fallback: u32) -> nabu_core::Result<u32> {
    use std::os::unix::fs::PermissionsExt;

    match fs::metadata(path) {
        Ok(metadata) => Ok(metadata.permissions().mode() & 0o777),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(fallback),
        Err(source) => Err(Error::Io {
            path: path.to_path_buf(),
            source,
        }),
    }
}

#[cfg(not(unix))]
fn file_mode_or(_path: &Path, fallback: u32) -> nabu_core::Result<u32> {
    Ok(fallback)
}

#[cfg(unix)]
fn chmod_path(path: &Path, mode: u32) -> nabu_core::Result<()> {
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
fn chmod_path(_path: &Path, _mode: u32) -> nabu_core::Result<()> {
    Ok(())
}
