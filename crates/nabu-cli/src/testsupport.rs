//! Shared test-only helpers.
//!
//! `ENV_LOCK` serializes every test that mutates process-global environment
//! variables (`CODEX_HOME`/`HOME`/`CLAUDE_CONFIG_DIR`/`OPENCODE_CONFIG_DIR`/
//! `XDG_CONFIG_HOME`). `std::env::set_var`/`remove_var` are process-wide, so
//! these tests cannot run in parallel; every env-mutating test acquires this one
//! lock. As the test suite is split across modules, each module's tests import
//! these helpers rather than re-deriving a private lock.

use std::path::Path;
use std::sync::Mutex;

pub(crate) static ENV_LOCK: Mutex<()> = Mutex::new(());

#[cfg(unix)]
pub(crate) fn set_mode(path: &Path, mode: u32) {
    use std::os::unix::fs::PermissionsExt;

    let mut permissions = std::fs::metadata(path).unwrap().permissions();
    permissions.set_mode(mode);
    std::fs::set_permissions(path, permissions).unwrap();
}

#[cfg(not(unix))]
pub(crate) fn set_mode(_path: &Path, _mode: u32) {}

#[cfg(unix)]
pub(crate) fn file_mode(path: &Path) -> u32 {
    use std::os::unix::fs::PermissionsExt;

    std::fs::metadata(path).unwrap().permissions().mode() & 0o777
}

#[cfg(not(unix))]
pub(crate) fn file_mode(_path: &Path) -> u32 {
    0
}

pub(crate) struct EnvGuard {
    previous: Vec<(&'static str, Option<std::ffi::OsString>)>,
}

impl EnvGuard {
    pub(crate) fn set<const N: usize>(values: [(&'static str, &std::ffi::OsStr); N]) -> Self {
        let previous = values
            .iter()
            .map(|(key, _)| (*key, std::env::var_os(key)))
            .collect::<Vec<_>>();
        for (key, value) in values {
            std::env::set_var(key, value);
        }
        Self { previous }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (key, value) in self.previous.drain(..) {
            if let Some(value) = value {
                std::env::set_var(key, value);
            } else {
                std::env::remove_var(key);
            }
        }
    }
}
