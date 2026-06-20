//! Home/store path resolution and filesystem-permission helpers.

use crate::{sanitize_session_id, Error, Result, Tool};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

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

pub(crate) fn create_dir_0700(path: &Path) -> Result<()> {
    fs::create_dir_all(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    chmod(path, 0o700)
}

pub(crate) fn lock_path_for_raw_file(raw_file: &Path) -> PathBuf {
    raw_file.with_extension("jsonl.lock")
}

pub(crate) fn harness_home_for_raw_file(raw_file: &Path) -> PathBuf {
    raw_file
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

#[cfg(unix)]
pub(crate) fn chmod(path: &Path, mode: u32) -> Result<()> {
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
pub(crate) fn chmod(_path: &Path, _mode: u32) -> Result<()> {
    Ok(())
}

pub(crate) fn set_if_exists(path: &Path, mode: u32) -> Result<()> {
    if path.exists() {
        chmod(path, mode)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_raw_path_uses_tool_and_sanitized_session_id() {
        let path = canonical_raw_path(Path::new("/tmp/harness"), Tool::Claude, "a/b c");

        assert_eq!(
            path,
            PathBuf::from("/tmp/harness/raw/claude/claude_a_b_c.jsonl")
        );
    }
}
