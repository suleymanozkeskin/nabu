use nabu_core::{opencode_server_url, Error, Result, Tool};
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

pub use nabu_core as core;

const CLAUDE_HOOK_EVENTS: [&str; 12] = [
    "SessionStart",
    "UserPromptSubmit",
    "MessageDisplay",
    "PreToolUse",
    "PostToolUse",
    "PostToolUseFailure",
    "PostToolBatch",
    "PreCompact",
    "PostCompact",
    "Stop",
    "StopFailure",
    "SessionEnd",
];
const OPENCODE_PLUGIN: &str = include_str!("../templates/harness-history.ts");
#[cfg(test)]
const CODEX_HOOKS_REFERENCE: &str = include_str!("../templates/codex-hooks.json");
#[cfg(test)]
const CLAUDE_SETTINGS_FRAGMENT_REFERENCE: &str =
    include_str!("../templates/claude-settings.fragment.json");
const CODEX_HOOK_EVENTS: [&str; 9] = [
    "SessionStart",
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "PreCompact",
    "PostCompact",
    "SubagentStart",
    "SubagentStop",
    "Stop",
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ConfigChangeReport {
    pub tool: Tool,
    pub target_path: PathBuf,
    pub changed: bool,
    pub dry_run: bool,
    pub summary: String,
    pub diff: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ClaudeStatus {
    pub settings_path: PathBuf,
    pub hooks_installed: bool,
    pub claude_installed: bool,
    pub storage_writable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OpenCodeStatus {
    pub plugin_path: PathBuf,
    pub plugin_installed: bool,
    pub config_status: String,
    pub server_url: Option<String>,
    pub reconciliation_enabled: bool,
    pub opencode_installed: bool,
    pub storage_writable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CodexStatus {
    pub hooks_path: PathBuf,
    pub hooks_installed: bool,
    pub codex_installed: bool,
    pub trust_guidance: String,
    pub storage_writable: bool,
}

pub fn install_claude(home: &Path, dry_run: bool) -> Result<ConfigChangeReport> {
    let settings_path = claude_settings_path()?;
    let before = read_settings_or_empty(&settings_path)?;
    let command = format!("nabu ingest hook --tool claude --home {}", home.display());
    let after = add_claude_hooks(before.clone(), &command);
    let changed = before != after;
    let diff = json_diff(&before, &after)?;

    if changed && !dry_run {
        if settings_path.exists() {
            backup_config(home, Tool::Claude, "install", &settings_path)?;
        } else if let Some(parent) = settings_path.parent() {
            fs::create_dir_all(parent).map_err(|source| Error::Io {
                path: parent.to_path_buf(),
                source,
            })?;
            chmod(parent, 0o700)?;
        }
        write_json_file(&settings_path, &after, 0o600)?;
    }

    Ok(ConfigChangeReport {
        tool: Tool::Claude,
        target_path: settings_path,
        changed,
        dry_run,
        summary: if dry_run {
            "dry-run: Claude Code hook settings diff only".to_string()
        } else if changed {
            "installed Claude Code hook settings".to_string()
        } else {
            "Claude Code hook settings already installed".to_string()
        },
        diff,
    })
}

pub fn uninstall_claude(home: &Path, dry_run: bool) -> Result<ConfigChangeReport> {
    let settings_path = claude_settings_path()?;
    let before = read_settings_or_empty(&settings_path)?;
    let after = remove_claude_hooks(before.clone());
    let changed = before != after;
    let diff = json_diff(&before, &after)?;

    if changed && !dry_run {
        backup_config(home, Tool::Claude, "uninstall", &settings_path)?;
        write_json_file(&settings_path, &after, 0o600)?;
    }

    Ok(ConfigChangeReport {
        tool: Tool::Claude,
        target_path: settings_path,
        changed,
        dry_run,
        summary: if dry_run {
            "dry-run: Claude Code hook removal diff only".to_string()
        } else if changed {
            "removed Claude Code hook settings".to_string()
        } else {
            "Claude Code hook settings were not installed".to_string()
        },
        diff,
    })
}

pub fn install_opencode(home: &Path, dry_run: bool) -> Result<ConfigChangeReport> {
    let plugin_path = opencode_plugin_path()?;
    let before = if plugin_path.exists() {
        fs::read_to_string(&plugin_path).map_err(|source| Error::Io {
            path: plugin_path.clone(),
            source,
        })?
    } else {
        String::new()
    };
    let after = OPENCODE_PLUGIN.to_string();
    let changed = before != after;
    let diff = text_diff(&before, &after);

    if changed && !dry_run {
        if plugin_path.exists() {
            backup_config(home, Tool::Opencode, "install", &plugin_path)?;
        }
        if let Some(parent) = plugin_path.parent() {
            fs::create_dir_all(parent).map_err(|source| Error::Io {
                path: parent.to_path_buf(),
                source,
            })?;
            chmod(parent, 0o700)?;
        }
        write_text_file(&plugin_path, &after, 0o644)?;
    }

    Ok(ConfigChangeReport {
        tool: Tool::Opencode,
        target_path: plugin_path,
        changed,
        dry_run,
        summary: if dry_run {
            "dry-run: OpenCode plugin file diff only".to_string()
        } else if changed {
            "installed OpenCode Harness plugin".to_string()
        } else {
            "OpenCode Harness plugin already installed".to_string()
        },
        diff,
    })
}

pub fn install_codex(home: &Path, dry_run: bool) -> Result<ConfigChangeReport> {
    let hooks_path = codex_hooks_path()?;
    let before = read_settings_or_empty(&hooks_path)?;
    let command = format!("nabu ingest hook --tool codex --home {}", home.display());
    let after = add_codex_hooks(before.clone(), &command);
    let changed = before != after;
    let diff = json_diff(&before, &after)?;

    if changed && !dry_run {
        if hooks_path.exists() {
            backup_config(home, Tool::Codex, "install", &hooks_path)?;
        } else if let Some(parent) = hooks_path.parent() {
            fs::create_dir_all(parent).map_err(|source| Error::Io {
                path: parent.to_path_buf(),
                source,
            })?;
            chmod(parent, 0o700)?;
        }
        write_json_file(&hooks_path, &after, 0o600)?;
    }

    Ok(ConfigChangeReport {
        tool: Tool::Codex,
        target_path: hooks_path,
        changed,
        dry_run,
        summary: if dry_run {
            "dry-run: Codex hooks.json diff only".to_string()
        } else if changed {
            "installed Codex hook settings".to_string()
        } else {
            "Codex hook settings already installed".to_string()
        },
        diff,
    })
}

pub fn uninstall_codex(home: &Path, dry_run: bool) -> Result<ConfigChangeReport> {
    let hooks_path = codex_hooks_path()?;
    let before = read_settings_or_empty(&hooks_path)?;
    let after = remove_codex_hooks(before.clone());
    let changed = before != after;
    let diff = json_diff(&before, &after)?;

    if changed && !dry_run {
        backup_config(home, Tool::Codex, "uninstall", &hooks_path)?;
        write_json_file(&hooks_path, &after, 0o600)?;
    }

    Ok(ConfigChangeReport {
        tool: Tool::Codex,
        target_path: hooks_path,
        changed,
        dry_run,
        summary: if dry_run {
            "dry-run: Codex hook removal diff only".to_string()
        } else if changed {
            "removed Codex hook settings".to_string()
        } else {
            "Codex hook settings were not installed".to_string()
        },
        diff,
    })
}

pub fn uninstall_opencode(home: &Path, dry_run: bool) -> Result<ConfigChangeReport> {
    let plugin_path = opencode_plugin_path()?;
    let before = if plugin_path.exists() {
        fs::read_to_string(&plugin_path).map_err(|source| Error::Io {
            path: plugin_path.clone(),
            source,
        })?
    } else {
        String::new()
    };
    let changed = plugin_path.exists();
    let diff = text_diff(&before, "");

    if changed && !dry_run {
        backup_config(home, Tool::Opencode, "uninstall", &plugin_path)?;
        fs::remove_file(&plugin_path).map_err(|source| Error::Io {
            path: plugin_path.clone(),
            source,
        })?;
    }

    Ok(ConfigChangeReport {
        tool: Tool::Opencode,
        target_path: plugin_path,
        changed,
        dry_run,
        summary: if dry_run {
            "dry-run: OpenCode plugin removal diff only".to_string()
        } else if changed {
            "removed OpenCode Harness plugin".to_string()
        } else {
            "OpenCode Harness plugin was not installed".to_string()
        },
        diff,
    })
}

pub fn claude_status(home: &Path) -> Result<ClaudeStatus> {
    let settings_path = claude_settings_path()?;
    let settings = read_settings_or_empty(&settings_path)?;
    Ok(ClaudeStatus {
        settings_path,
        hooks_installed: settings_contains_harness_claude_hooks(&settings),
        claude_installed: command_in_path("claude"),
        storage_writable: home.join("raw").join("claude").is_dir(),
    })
}

pub fn opencode_status(home: &Path) -> Result<OpenCodeStatus> {
    let plugin_path = opencode_plugin_path()?;
    let server_url = opencode_server_url(home)?;
    Ok(OpenCodeStatus {
        plugin_installed: plugin_path.is_file(),
        plugin_path,
        config_status: "user-level plugin file".to_string(),
        reconciliation_enabled: server_url.is_some(),
        server_url,
        opencode_installed: command_in_path("opencode"),
        storage_writable: home.join("raw").join("opencode").is_dir(),
    })
}

pub fn codex_status(home: &Path) -> Result<CodexStatus> {
    let hooks_path = codex_hooks_path()?;
    let settings = read_settings_or_empty(&hooks_path)?;
    Ok(CodexStatus {
        hooks_installed: settings_contains_harness_codex_hooks(&settings),
        hooks_path,
        codex_installed: command_in_path("codex"),
        trust_guidance: "Codex compatibility mode captures turn-boundary hooks and reconciles transcripts; assistant deltas require streaming mode.".to_string(),
        storage_writable: home.join("raw").join("codex").is_dir(),
    })
}

fn claude_settings_path() -> Result<PathBuf> {
    if let Some(config_dir) = env::var_os("CLAUDE_CONFIG_DIR") {
        return Ok(PathBuf::from(config_dir).join("settings.json"));
    }
    let Some(home) = env::var_os("HOME") else {
        return Err(Error::HomeUnavailable);
    };
    Ok(PathBuf::from(home).join(".claude").join("settings.json"))
}

fn codex_hooks_path() -> Result<PathBuf> {
    if let Some(codex_home) = env::var_os("CODEX_HOME") {
        return Ok(PathBuf::from(codex_home).join("hooks.json"));
    }
    let Some(home) = env::var_os("HOME") else {
        return Err(Error::HomeUnavailable);
    };
    Ok(PathBuf::from(home).join(".codex").join("hooks.json"))
}

fn opencode_plugin_path() -> Result<PathBuf> {
    if let Some(config_dir) = env::var_os("OPENCODE_CONFIG_DIR") {
        return Ok(PathBuf::from(config_dir)
            .join("plugins")
            .join("harness-history.ts"));
    }
    if let Some(config_home) = env::var_os("XDG_CONFIG_HOME") {
        return Ok(PathBuf::from(config_home)
            .join("opencode")
            .join("plugins")
            .join("harness-history.ts"));
    }
    let Some(home) = env::var_os("HOME") else {
        return Err(Error::HomeUnavailable);
    };
    Ok(PathBuf::from(home)
        .join(".config")
        .join("opencode")
        .join("plugins")
        .join("harness-history.ts"))
}

fn read_settings_or_empty(path: &Path) -> Result<Value> {
    if !path.exists() {
        return Ok(json!({}));
    }
    let content = fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(serde_json::from_str(&content)?)
}

fn add_claude_hooks(mut settings: Value, command: &str) -> Value {
    ensure_object(&mut settings);
    let hooks = settings
        .as_object_mut()
        .expect("settings object")
        .entry("hooks")
        .or_insert_with(|| json!({}));
    ensure_object(hooks);
    let hooks_object = hooks.as_object_mut().expect("hooks object");

    for event in CLAUDE_HOOK_EVENTS {
        let entries = hooks_object
            .entry(event.to_string())
            .or_insert_with(|| Value::Array(Vec::new()));
        ensure_array(entries);
        let entries_array = entries.as_array_mut().expect("hook entries");
        let already_present = entries_array.iter().any(|entry| {
            entry
                .pointer("/hooks/0/command")
                .and_then(Value::as_str)
                .map(|existing| existing == command)
                .unwrap_or(false)
        });
        if !already_present {
            entries_array.push(json!({
                "hooks": [
                    {
                        "type": "command",
                        "command": command
                    }
                ]
            }));
        }
    }

    settings
}

fn add_codex_hooks(mut settings: Value, command: &str) -> Value {
    ensure_object(&mut settings);
    let hooks = settings
        .as_object_mut()
        .expect("settings object")
        .entry("hooks")
        .or_insert_with(|| json!({}));
    ensure_object(hooks);
    let hooks_object = hooks.as_object_mut().expect("hooks object");

    for event in CODEX_HOOK_EVENTS {
        let entries = hooks_object
            .entry(event.to_string())
            .or_insert_with(|| Value::Array(Vec::new()));
        ensure_array(entries);
        let entries_array = entries.as_array_mut().expect("hook entries");
        let already_present = entries_array.iter().any(|entry| {
            entry
                .get("command")
                .and_then(Value::as_str)
                .map(|existing| existing == command)
                .unwrap_or(false)
        });
        if !already_present {
            entries_array.push(json!({
                "type": "command",
                "command": command
            }));
        }
    }

    settings
}

fn remove_claude_hooks(mut settings: Value) -> Value {
    let Some(hooks_object) = settings.get_mut("hooks").and_then(Value::as_object_mut) else {
        return settings;
    };

    for event in CLAUDE_HOOK_EVENTS {
        if let Some(entries) = hooks_object.get_mut(event).and_then(Value::as_array_mut) {
            entries.retain(|entry| {
                !entry
                    .pointer("/hooks/0/command")
                    .and_then(Value::as_str)
                    // Binary-agnostic: also removes pre-rename `tupsharrum ingest ...` hooks.
                    .map(|command| command.contains("ingest hook --tool claude"))
                    .unwrap_or(false)
            });
        }
    }
    hooks_object.retain(|_, value| {
        value
            .as_array()
            .map(|entries| !entries.is_empty())
            .unwrap_or(true)
    });

    settings
}

fn remove_codex_hooks(mut settings: Value) -> Value {
    let Some(hooks_object) = settings.get_mut("hooks").and_then(Value::as_object_mut) else {
        return settings;
    };

    for event in CODEX_HOOK_EVENTS {
        if let Some(entries) = hooks_object.get_mut(event).and_then(Value::as_array_mut) {
            entries.retain(|entry| {
                !entry
                    .get("command")
                    .and_then(Value::as_str)
                    // Binary-agnostic: also removes pre-rename `tupsharrum ingest ...` hooks.
                    .map(|command| command.contains("ingest hook --tool codex"))
                    .unwrap_or(false)
            });
        }
    }
    hooks_object.retain(|_, value| {
        value
            .as_array()
            .map(|entries| !entries.is_empty())
            .unwrap_or(true)
    });

    settings
}

fn settings_contains_harness_claude_hooks(settings: &Value) -> bool {
    let Some(hooks_object) = settings.get("hooks").and_then(Value::as_object) else {
        return false;
    };
    CLAUDE_HOOK_EVENTS.iter().all(|event| {
        hooks_object
            .get(*event)
            .and_then(Value::as_array)
            .map(|entries| {
                entries.iter().any(|entry| {
                    entry
                        .pointer("/hooks/0/command")
                        .and_then(Value::as_str)
                        .map(|command| command.contains("nabu ingest hook --tool claude"))
                        .unwrap_or(false)
                })
            })
            .unwrap_or(false)
    })
}

fn settings_contains_harness_codex_hooks(settings: &Value) -> bool {
    let Some(hooks_object) = settings.get("hooks").and_then(Value::as_object) else {
        return false;
    };
    CODEX_HOOK_EVENTS.iter().all(|event| {
        hooks_object
            .get(*event)
            .and_then(Value::as_array)
            .map(|entries| {
                entries.iter().any(|entry| {
                    entry
                        .get("command")
                        .and_then(Value::as_str)
                        .map(|command| command.contains("nabu ingest hook --tool codex"))
                        .unwrap_or(false)
                })
            })
            .unwrap_or(false)
    })
}

fn command_in_path(command: &str) -> bool {
    let Some(paths) = env::var_os("PATH") else {
        return false;
    };
    env::split_paths(&paths).any(|path| path.join(command).is_file())
}

fn ensure_object(value: &mut Value) {
    if !value.is_object() {
        *value = json!({});
    }
}

fn ensure_array(value: &mut Value) {
    if !value.is_array() {
        *value = Value::Array(Vec::new());
    }
}

fn json_diff(before: &Value, after: &Value) -> Result<String> {
    Ok(format!(
        "--- before\n{}\n--- after\n{}\n",
        serde_json::to_string_pretty(before)?,
        serde_json::to_string_pretty(after)?
    ))
}

fn write_json_file(path: &Path, value: &Value, mode: u32) -> Result<()> {
    let final_mode = file_mode_or(path, mode)?;
    let content = serde_json::to_vec_pretty(value)?;
    fs::write(path, content).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    chmod(path, final_mode)
}

fn write_text_file(path: &Path, content: &str, mode: u32) -> Result<()> {
    let final_mode = file_mode_or(path, mode)?;
    fs::write(path, content).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    chmod(path, final_mode)
}

fn text_diff(before: &str, after: &str) -> String {
    format!("--- before\n{before}\n--- after\n{after}\n")
}

fn backup_config(home: &Path, tool: Tool, operation: &str, original_path: &Path) -> Result<()> {
    let content = fs::read(original_path).map_err(|source| Error::Io {
        path: original_path.to_path_buf(),
        source,
    })?;
    let now = OffsetDateTime::now_utc();
    let created_at = now.format(&Rfc3339)?;
    let stamp = backup_stamp(now);
    let hash = sha256_hex(&content);
    let backup_path = original_path.with_file_name(format!(
        "{}.nabu-backup.{}.{}.bak",
        original_path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("settings.json"),
        stamp,
        &hash[..8]
    ));
    fs::write(&backup_path, &content).map_err(|source| Error::Io {
        path: backup_path.clone(),
        source,
    })?;
    chmod(&backup_path, 0o600)?;

    let backups_dir = home.join("backups");
    fs::create_dir_all(&backups_dir).map_err(|source| Error::Io {
        path: backups_dir.clone(),
        source,
    })?;
    chmod(&backups_dir, 0o700)?;
    let manifest_path = backups_dir.join("manifest.jsonl");
    let record = json!({
        "created_at": created_at,
        "tool": tool.as_str(),
        "operation": operation,
        "original_path": original_path.display().to_string(),
        "backup_path": backup_path.display().to_string(),
        "sha256": sha256_hex(&content)
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
    chmod(&manifest_path, 0o600)
}

fn sha256_hex(content: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content);
    hex::encode(hasher.finalize())
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

#[cfg(unix)]
fn file_mode_or(path: &Path, fallback: u32) -> Result<u32> {
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
fn file_mode_or(_path: &Path, fallback: u32) -> Result<u32> {
    Ok(fallback)
}

#[cfg(unix)]
fn chmod(path: &Path, mode: u32) -> Result<()> {
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
fn chmod(_path: &Path, _mode: u32) -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::{OsStr, OsString};
    use std::sync::Mutex;
    use tempfile::tempdir;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn opencode_plugin_declares_required_events_and_fail_open_logging() {
        for event_name in [
            "message.updated",
            "message.part.updated",
            "message.removed",
            "message.part.removed",
            "session.created",
            "session.updated",
            "session.compacted",
            "session.idle",
            "session.error",
            "tool.execute.before",
            "tool.execute.after",
            "command.executed",
            "file.edited",
        ] {
            assert!(OPENCODE_PLUGIN.contains(event_name), "{event_name}");
        }
        assert!(OPENCODE_PLUGIN.contains("catch (error)"));
        assert!(OPENCODE_PLUGIN.contains("console.error"));
        assert!(OPENCODE_PLUGIN.contains("nabu"));
        assert!(OPENCODE_PLUGIN.contains("ingest"));
        assert!(OPENCODE_PLUGIN.contains("--tool"));
        assert!(OPENCODE_PLUGIN.contains("opencode"));
    }

    #[test]
    fn live_adapter_event_sets_match_feature_set() {
        assert_eq!(
            CLAUDE_HOOK_EVENTS,
            [
                "SessionStart",
                "UserPromptSubmit",
                "MessageDisplay",
                "PreToolUse",
                "PostToolUse",
                "PostToolUseFailure",
                "PostToolBatch",
                "PreCompact",
                "PostCompact",
                "Stop",
                "StopFailure",
                "SessionEnd",
            ]
        );
        assert_eq!(
            CODEX_HOOK_EVENTS,
            [
                "SessionStart",
                "UserPromptSubmit",
                "PreToolUse",
                "PostToolUse",
                "PreCompact",
                "PostCompact",
                "SubagentStart",
                "SubagentStop",
                "Stop",
            ]
        );
        for event_name in CLAUDE_HOOK_EVENTS {
            assert!(CLAUDE_SETTINGS_FRAGMENT_REFERENCE.contains(event_name));
        }
        for event_name in CODEX_HOOK_EVENTS {
            assert!(CODEX_HOOKS_REFERENCE.contains(event_name));
        }
    }

    #[test]
    fn committed_hook_fragments_match_installer_output() {
        let home = Path::new("/tmp/nabu-reference-home");
        let codex_command = format!("nabu ingest hook --tool codex --home {}", home.display());
        let claude_command = format!("nabu ingest hook --tool claude --home {}", home.display());
        let rendered_codex: Value = serde_json::from_str(
            &CODEX_HOOKS_REFERENCE.replace("__NABU_HOME__", &home.display().to_string()),
        )
        .unwrap();
        let rendered_claude: Value = serde_json::from_str(
            &CLAUDE_SETTINGS_FRAGMENT_REFERENCE
                .replace("__NABU_HOME__", &home.display().to_string()),
        )
        .unwrap();

        assert_eq!(add_codex_hooks(json!({}), &codex_command), rendered_codex);
        assert_eq!(
            add_claude_hooks(json!({}), &claude_command),
            rendered_claude
        );
        assert_eq!(
            rendered_codex
                .pointer("/hooks")
                .and_then(Value::as_object)
                .unwrap()
                .len(),
            CODEX_HOOK_EVENTS.len()
        );
        assert_eq!(
            rendered_claude
                .pointer("/hooks")
                .and_then(Value::as_object)
                .unwrap()
                .len(),
            CLAUDE_HOOK_EVENTS.len()
        );
    }

    #[test]
    fn native_install_uninstall_preserves_configs_and_records_backups() {
        let _guard = ENV_LOCK.lock().unwrap();
        let temp = tempdir().unwrap();
        let harness_home = temp.path().join("raven");
        let fake_home = temp.path().join("home");
        let codex_home = temp.path().join("codex");
        let claude_config = temp.path().join("claude");
        let opencode_config = temp.path().join("opencode");
        fs::create_dir_all(&fake_home).unwrap();
        fs::create_dir_all(&codex_home).unwrap();
        fs::create_dir_all(&claude_config).unwrap();
        fs::create_dir_all(&opencode_config).unwrap();
        nabu_core::init_home(&harness_home).unwrap();

        let codex_hooks = codex_home.join("hooks.json");
        let claude_settings = claude_config.join("settings.json");
        let opencode_plugin = opencode_config.join("plugins/harness-history.ts");
        fs::write(
            &codex_hooks,
            r#"{"theme":"dark","hooks":{"Stop":[{"type":"command","command":"echo keep-codex"}]}}"#,
        )
        .unwrap();
        fs::write(
            &claude_settings,
            r#"{"theme":"dark","hooks":{"Stop":[{"hooks":[{"type":"command","command":"echo keep-claude"}]}]}}"#,
        )
        .unwrap();
        set_mode(&codex_hooks, 0o640);
        set_mode(&claude_settings, 0o640);

        let env_guard = EnvGuard::set([
            ("HOME", fake_home.as_os_str()),
            ("CODEX_HOME", codex_home.as_os_str()),
            ("CLAUDE_CONFIG_DIR", claude_config.as_os_str()),
            ("OPENCODE_CONFIG_DIR", opencode_config.as_os_str()),
        ]);

        let before_codex = fs::read_to_string(&codex_hooks).unwrap();
        let before_claude = fs::read_to_string(&claude_settings).unwrap();
        let dry_reports = [
            install_codex(&harness_home, true).unwrap(),
            install_claude(&harness_home, true).unwrap(),
            install_opencode(&harness_home, true).unwrap(),
        ];
        assert!(dry_reports.iter().all(|report| report.dry_run));
        assert_eq!(fs::read_to_string(&codex_hooks).unwrap(), before_codex);
        assert_eq!(fs::read_to_string(&claude_settings).unwrap(), before_claude);
        assert!(!opencode_plugin.exists());
        assert!(!harness_home.join("backups/manifest.jsonl").exists());

        for report in [
            install_codex(&harness_home, false).unwrap(),
            install_claude(&harness_home, false).unwrap(),
            install_opencode(&harness_home, false).unwrap(),
        ] {
            assert!(report.changed);
        }

        let codex: Value =
            serde_json::from_str(&fs::read_to_string(&codex_hooks).unwrap()).unwrap();
        assert_eq!(codex["theme"], "dark");
        assert!(codex.to_string().contains("keep-codex"));
        assert!(settings_contains_harness_codex_hooks(&codex));
        let claude: Value =
            serde_json::from_str(&fs::read_to_string(&claude_settings).unwrap()).unwrap();
        assert_eq!(claude["theme"], "dark");
        assert!(claude.to_string().contains("keep-claude"));
        assert!(settings_contains_harness_claude_hooks(&claude));
        assert!(opencode_plugin.is_file());
        assert_eq!(file_mode(&codex_hooks), 0o640);
        assert_eq!(file_mode(&claude_settings), 0o640);
        assert_eq!(file_mode(&opencode_plugin), 0o644);

        let manifest_path = harness_home.join("backups/manifest.jsonl");
        let install_manifest = fs::read_to_string(&manifest_path).unwrap();
        assert_eq!(install_manifest.lines().count(), 2);
        for line in install_manifest.lines() {
            let record: Value = serde_json::from_str(line).unwrap();
            assert_eq!(record["operation"], "install");
            let backup_path = PathBuf::from(record["backup_path"].as_str().unwrap());
            assert!(backup_path.is_file());
            assert_eq!(file_mode(&backup_path), 0o600);
            assert!(backup_path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .contains(".nabu-backup."));
        }

        for report in [
            install_codex(&harness_home, false).unwrap(),
            install_claude(&harness_home, false).unwrap(),
            install_opencode(&harness_home, false).unwrap(),
        ] {
            assert!(!report.changed);
        }
        assert_eq!(
            fs::read_to_string(&manifest_path).unwrap().lines().count(),
            2
        );

        for report in [
            uninstall_codex(&harness_home, false).unwrap(),
            uninstall_claude(&harness_home, false).unwrap(),
            uninstall_opencode(&harness_home, false).unwrap(),
        ] {
            assert!(report.changed);
        }

        let codex: Value =
            serde_json::from_str(&fs::read_to_string(&codex_hooks).unwrap()).unwrap();
        assert_eq!(codex["theme"], "dark");
        assert!(codex.to_string().contains("keep-codex"));
        assert!(!settings_contains_harness_codex_hooks(&codex));
        let claude: Value =
            serde_json::from_str(&fs::read_to_string(&claude_settings).unwrap()).unwrap();
        assert_eq!(claude["theme"], "dark");
        assert!(claude.to_string().contains("keep-claude"));
        assert!(!settings_contains_harness_claude_hooks(&claude));
        assert!(!opencode_plugin.exists());
        assert_eq!(
            fs::read_to_string(&manifest_path).unwrap().lines().count(),
            5
        );

        drop(env_guard);
    }

    struct EnvGuard {
        old_values: Vec<(&'static str, Option<OsString>)>,
    }

    impl EnvGuard {
        fn set<const N: usize>(values: [(&'static str, &OsStr); N]) -> Self {
            let mut old_values = Vec::new();
            for (key, value) in values {
                old_values.push((key, env::var_os(key)));
                env::set_var(key, value);
            }
            Self { old_values }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (key, value) in self.old_values.drain(..).rev() {
                if let Some(value) = value {
                    env::set_var(key, value);
                } else {
                    env::remove_var(key);
                }
            }
        }
    }

    #[cfg(unix)]
    fn set_mode(path: &Path, mode: u32) {
        use std::os::unix::fs::PermissionsExt;

        let mut permissions = fs::metadata(path).unwrap().permissions();
        permissions.set_mode(mode);
        fs::set_permissions(path, permissions).unwrap();
    }

    #[cfg(not(unix))]
    fn set_mode(_path: &Path, _mode: u32) {}

    #[cfg(unix)]
    fn file_mode(path: &Path) -> u32 {
        use std::os::unix::fs::PermissionsExt;

        fs::metadata(path).unwrap().permissions().mode() & 0o777
    }

    #[cfg(not(unix))]
    fn file_mode(_path: &Path) -> u32 {
        0
    }
}
