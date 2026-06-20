//! Per-tool filesystem layout for the external agent tools.
//!
//! Codex, Claude Code, and OpenCode each store their session transcripts and
//! their MCP config in tool-specific locations with tool-specific environment
//! precedence. Rather than scatter that knowledge across `codex_*`/`claude_*`/
//! `opencode_*` free functions — which forces every caller to restate the tool
//! once in a `match` arm and again in the function name — the layout is attached
//! to `Tool` via the [`ToolLayout`] extension trait. Callers ask the tool
//! directly (`tool.mcp_config_path()`), and adding a tool variant makes the
//! exhaustive `match` inside each method a compile error until its layout is
//! supplied.

use nabu_core::{Error, Tool};
use std::path::PathBuf;

/// Filesystem layout for an external agent tool: where it keeps its session
/// transcripts (for backfill discovery) and its MCP config file (for
/// install/uninstall).
pub(crate) trait ToolLayout {
    /// Directories to scan for this tool's session transcripts. Codex keeps both
    /// live and archived sessions, so it returns two roots; the others return
    /// one. Order is significant and preserved by callers.
    fn transcript_roots(self) -> nabu_core::Result<Vec<PathBuf>>;

    /// The config file into which the `nabu` MCP server is registered.
    fn mcp_config_path(self) -> nabu_core::Result<PathBuf>;
}

impl ToolLayout for Tool {
    fn transcript_roots(self) -> nabu_core::Result<Vec<PathBuf>> {
        match self {
            Tool::Codex => {
                let home = codex_home_dir()?;
                Ok(vec![home.join("sessions"), home.join("archived_sessions")])
            }
            Tool::Claude => {
                let projects = if let Some(config_dir) = std::env::var_os("CLAUDE_CONFIG_DIR") {
                    PathBuf::from(config_dir).join("projects")
                } else {
                    home_dir()?.join(".claude").join("projects")
                };
                Ok(vec![projects])
            }
            Tool::Opencode => Ok(vec![home_dir()?
                .join(".local")
                .join("share")
                .join("opencode")]),
        }
    }

    fn mcp_config_path(self) -> nabu_core::Result<PathBuf> {
        match self {
            // Codex stores its config under the same CODEX_HOME base as its
            // sessions, so the path is the home dir plus config.toml.
            Tool::Codex => Ok(codex_home_dir()?.join("config.toml")),
            Tool::Claude => Ok(home_dir()?.join(".claude.json")),
            Tool::Opencode => {
                if let Some(config_dir) = std::env::var_os("OPENCODE_CONFIG_DIR") {
                    Ok(PathBuf::from(config_dir).join("opencode.json"))
                } else if let Some(config_home) = std::env::var_os("XDG_CONFIG_HOME") {
                    Ok(PathBuf::from(config_home)
                        .join("opencode")
                        .join("opencode.json"))
                } else {
                    Ok(home_dir()?
                        .join(".config")
                        .join("opencode")
                        .join("opencode.json"))
                }
            }
        }
    }
}

/// `$HOME`, or [`Error::HomeUnavailable`] when it is unset.
fn home_dir() -> nabu_core::Result<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .ok_or(Error::HomeUnavailable)
}

/// Codex's home: `$CODEX_HOME`, else `$HOME/.codex`.
fn codex_home_dir() -> nabu_core::Result<PathBuf> {
    if let Some(codex_home) = std::env::var_os("CODEX_HOME") {
        return Ok(PathBuf::from(codex_home));
    }
    Ok(home_dir()?.join(".codex"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testsupport::{EnvGuard, ENV_LOCK};
    use tempfile::tempdir;

    #[test]
    fn codex_layout_shares_one_codex_home_base() {
        let _guard = ENV_LOCK.lock().unwrap();
        let temp = tempdir().unwrap();
        let codex_home = temp.path().join("codex");
        let env = EnvGuard::set([
            ("CODEX_HOME", codex_home.as_os_str()),
            ("HOME", temp.path().as_os_str()),
        ]);

        let roots = Tool::Codex.transcript_roots().unwrap();
        assert_eq!(
            roots,
            vec![
                codex_home.join("sessions"),
                codex_home.join("archived_sessions")
            ]
        );
        // The fold: config.toml resolves under the same CODEX_HOME base.
        assert_eq!(
            Tool::Codex.mcp_config_path().unwrap(),
            codex_home.join("config.toml")
        );
        drop(env);
    }

    #[test]
    fn claude_config_path_is_home_dotfile_independent_of_config_dir() {
        let _guard = ENV_LOCK.lock().unwrap();
        let temp = tempdir().unwrap();
        let config_dir = temp.path().join("claude-config");
        let env = EnvGuard::set([
            ("HOME", temp.path().as_os_str()),
            ("CLAUDE_CONFIG_DIR", config_dir.as_os_str()),
        ]);

        // Transcripts honor CLAUDE_CONFIG_DIR ...
        assert_eq!(
            Tool::Claude.transcript_roots().unwrap(),
            vec![config_dir.join("projects")]
        );
        // ... but the MCP config is always $HOME/.claude.json.
        assert_eq!(
            Tool::Claude.mcp_config_path().unwrap(),
            temp.path().join(".claude.json")
        );
        drop(env);
    }

    #[test]
    fn opencode_config_path_prefers_config_dir_then_xdg() {
        let _guard = ENV_LOCK.lock().unwrap();
        let temp = tempdir().unwrap();
        let config_dir = temp.path().join("oc");
        let env = EnvGuard::set([
            ("HOME", temp.path().as_os_str()),
            ("OPENCODE_CONFIG_DIR", config_dir.as_os_str()),
        ]);

        assert_eq!(
            Tool::Opencode.mcp_config_path().unwrap(),
            config_dir.join("opencode.json")
        );
        assert_eq!(
            Tool::Opencode.transcript_roots().unwrap(),
            vec![temp.path().join(".local").join("share").join("opencode")]
        );
        drop(env);
    }
}
