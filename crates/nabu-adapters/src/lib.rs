//! Agent configuration adapters for the `nabu` CLI: install and remove the
//! `nabu` hooks/MCP entries in Claude, Codex, and OpenCode config files.
//!
//! This crate is published only so the `nabu` binary (the `nabu-cli` crate)
//! resolves its dependencies. It is not a stable public API — items may change
//! or be removed in any release with no semver guarantee. Depend on the `nabu`
//! CLI, not on this crate directly.
#![doc(hidden)]

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
    /// `Some` when the settings file exists but is not valid JSON; the file is
    /// left untouched and `hooks_installed` is reported as `false`.
    pub parse_error: Option<String>,
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
    /// `Some` when the hooks file exists but is not valid JSON; the file is left
    /// untouched and `hooks_installed` is reported as `false`.
    pub parse_error: Option<String>,
}

pub fn install_claude(home: &Path, dry_run: bool) -> Result<ConfigChangeReport> {
    let settings_path = claude_settings_path()?;
    let before = read_settings_or_empty(&settings_path)?;
    let command = format!(
        "nabu ingest hook --tool claude --home {}",
        shell_single_quote(home)
    );
    let after = add_claude_hooks(before.clone(), &command)?;
    let changed = before != after;
    let diff = claude_hook_change_summary(&before, &after);

    if changed && !dry_run {
        if settings_path.exists() {
            backup_config(home, Tool::Claude, "install", &settings_path)?;
        } else if let Some(parent) = settings_path.parent() {
            create_dir_all_owned(parent)?;
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
    let diff = claude_hook_change_summary(&before, &after);

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
            create_dir_all_owned(parent)?;
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
    let command = format!(
        "nabu ingest hook --tool codex --home {}",
        shell_single_quote(home)
    );
    let after = add_codex_hooks(before.clone(), &command)?;
    let changed = before != after;
    let diff = codex_hook_change_summary(&before, &after);

    if changed && !dry_run {
        if hooks_path.exists() {
            backup_config(home, Tool::Codex, "install", &hooks_path)?;
        } else if let Some(parent) = hooks_path.parent() {
            create_dir_all_owned(parent)?;
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
    let diff = codex_hook_change_summary(&before, &after);

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
    let (hooks_installed, parse_error) = match read_settings_or_empty(&settings_path) {
        Ok(settings) => (settings_contains_harness_claude_hooks(&settings), None),
        Err(Error::Json(error)) => (false, Some(error.to_string())),
        Err(other) => return Err(other),
    };
    Ok(ClaudeStatus {
        settings_path,
        hooks_installed,
        claude_installed: command_in_path("claude"),
        storage_writable: home.join("raw").join("claude").is_dir(),
        parse_error,
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
    let (hooks_installed, parse_error) = match read_settings_or_empty(&hooks_path) {
        Ok(settings) => (settings_contains_harness_codex_hooks(&settings), None),
        Err(Error::Json(error)) => (false, Some(error.to_string())),
        Err(other) => return Err(other),
    };
    Ok(CodexStatus {
        hooks_installed,
        hooks_path,
        codex_installed: command_in_path("codex"),
        trust_guidance: "Codex compatibility mode captures turn-boundary hooks and reconciles transcripts; assistant deltas require streaming mode.".to_string(),
        storage_writable: home.join("raw").join("codex").is_dir(),
        parse_error,
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

// Binary-agnostic substrings: also match pre-rename `tupsharrum ingest ...` hooks
// so an upgrade uninstall leaves no orphan.
const CLAUDE_HOOK_MARKER: &str = "ingest hook --tool claude";
const CODEX_HOOK_MARKER: &str = "ingest hook --tool codex";

fn add_claude_hooks(mut settings: Value, command: &str) -> Result<Value> {
    require_object(&settings, "settings")?;
    let hooks = settings
        .as_object_mut()
        .expect("settings object")
        .entry("hooks")
        .or_insert_with(|| json!({}));
    require_object(hooks, "hooks")?;
    let hooks_object = hooks.as_object_mut().expect("hooks object");

    for event in CLAUDE_HOOK_EVENTS {
        let entries = hooks_object
            .entry(event.to_string())
            .or_insert_with(|| Value::Array(Vec::new()));
        require_event_array(entries, event)?;
        let entries_array = entries.as_array_mut().expect("hook entries");
        // Replace-then-add: prune stale nabu hooks (marker match, different
        // command — e.g. pre-quoting or previous --home installs) so upgrades
        // converge to exactly one canonical hook instead of stacking duplicates.
        prune_claude_hook_entries(entries_array, |hook| {
            hook_command_contains(hook, CLAUDE_HOOK_MARKER) && hook_command(hook) != Some(command)
        });
        // Claude allows multiple inner hooks per entry; the nabu command is
        // present if any inner hook of any entry carries it.
        let already_present = entries_array
            .iter()
            .flat_map(claude_inner_hooks)
            .any(|hook| hook_command(hook) == Some(command));
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

    Ok(settings)
}

fn add_codex_hooks(mut settings: Value, command: &str) -> Result<Value> {
    require_object(&settings, "settings")?;
    let hooks = settings
        .as_object_mut()
        .expect("settings object")
        .entry("hooks")
        .or_insert_with(|| json!({}));
    require_object(hooks, "hooks")?;
    let hooks_object = hooks.as_object_mut().expect("hooks object");

    for event in CODEX_HOOK_EVENTS {
        let entries = hooks_object
            .entry(event.to_string())
            .or_insert_with(|| Value::Array(Vec::new()));
        require_event_array(entries, event)?;
        let entries_array = entries.as_array_mut().expect("hook entries");
        // Replace-then-add: drop stale nabu entries (marker match, different
        // command) so upgrades converge to exactly one canonical hook.
        entries_array.retain(|entry| {
            !(hook_command_contains(entry, CODEX_HOOK_MARKER)
                && hook_command(entry) != Some(command))
        });
        let already_present = entries_array
            .iter()
            .any(|entry| hook_command(entry) == Some(command));
        if !already_present {
            entries_array.push(json!({
                "type": "command",
                "command": command
            }));
        }
    }

    Ok(settings)
}

fn remove_claude_hooks(mut settings: Value) -> Value {
    let Some(hooks_object) = settings.get_mut("hooks").and_then(Value::as_object_mut) else {
        return settings;
    };

    let mut emptied_events: Vec<String> = Vec::new();
    for event in CLAUDE_HOOK_EVENTS {
        let Some(entries) = hooks_object.get_mut(event).and_then(Value::as_array_mut) else {
            continue;
        };
        let before_len = entries.len();
        prune_claude_hook_entries(entries, |hook| {
            hook_command_contains(hook, CLAUDE_HOOK_MARKER)
        });
        // Track only event keys our filtering emptied, so user-created empty
        // arrays (or keys we never managed) are preserved.
        if entries.is_empty() && before_len > 0 {
            emptied_events.push(event.to_string());
        }
    }
    for event in emptied_events {
        hooks_object.remove(&event);
    }

    settings
}

fn remove_codex_hooks(mut settings: Value) -> Value {
    let Some(hooks_object) = settings.get_mut("hooks").and_then(Value::as_object_mut) else {
        return settings;
    };

    let mut emptied_events: Vec<String> = Vec::new();
    for event in CODEX_HOOK_EVENTS {
        let Some(entries) = hooks_object.get_mut(event).and_then(Value::as_array_mut) else {
            continue;
        };
        let before_len = entries.len();
        entries.retain(|entry| !hook_command_contains(entry, CODEX_HOOK_MARKER));
        if entries.is_empty() && before_len > 0 {
            emptied_events.push(event.to_string());
        }
    }
    for event in emptied_events {
        hooks_object.remove(&event);
    }

    settings
}

fn settings_contains_harness_claude_hooks(settings: &Value) -> bool {
    let Some(hooks_object) = settings.get("hooks").and_then(Value::as_object) else {
        return false;
    };
    CLAUDE_HOOK_EVENTS
        .iter()
        .all(|event| claude_event_has_nabu_hook(hooks_object.get(*event)))
}

fn settings_contains_harness_codex_hooks(settings: &Value) -> bool {
    let Some(hooks_object) = settings.get("hooks").and_then(Value::as_object) else {
        return false;
    };
    CODEX_HOOK_EVENTS
        .iter()
        .all(|event| codex_event_has_nabu_hook(hooks_object.get(*event)))
}

/// Remove inner hooks matching `is_stale` from each Claude entry, keeping
/// co-located user hooks; drop an entry only when the pruning emptied its
/// `hooks` array. Entries without a `hooks` array are not ours — left untouched.
fn prune_claude_hook_entries(entries: &mut Vec<Value>, is_stale: impl Fn(&Value) -> bool) {
    entries.retain_mut(|entry| {
        let Some(inner) = entry.get_mut("hooks").and_then(Value::as_array_mut) else {
            return true;
        };
        let inner_before = inner.len();
        inner.retain(|hook| !is_stale(hook));
        !(inner.is_empty() && inner_before > 0)
    });
}

/// Inner hooks of a Claude entry (`{matcher?, hooks: [...]}`); empty slice when
/// the entry has no valid `hooks` array.
fn claude_inner_hooks(entry: &Value) -> &[Value] {
    entry
        .get("hooks")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[])
}

fn hook_command(hook: &Value) -> Option<&str> {
    hook.get("command").and_then(Value::as_str)
}

fn hook_command_contains(hook: &Value, marker: &str) -> bool {
    hook_command(hook).is_some_and(|command| command.contains(marker))
}

fn claude_event_has_nabu_hook(event: Option<&Value>) -> bool {
    event.and_then(Value::as_array).is_some_and(|entries| {
        entries
            .iter()
            .flat_map(claude_inner_hooks)
            .any(|hook| hook_command_contains(hook, CLAUDE_HOOK_MARKER))
    })
}

fn codex_event_has_nabu_hook(event: Option<&Value>) -> bool {
    event.and_then(Value::as_array).is_some_and(|entries| {
        entries
            .iter()
            .any(|entry| hook_command_contains(entry, CODEX_HOOK_MARKER))
    })
}

/// Summarize which Claude hook events gained, lost, or replaced the nabu hook.
/// Reports only nabu's own commands per touched event key — never unrelated user
/// config (which may hold secrets).
fn claude_hook_change_summary(before: &Value, after: &Value) -> String {
    hook_change_summary(before, after, &CLAUDE_HOOK_EVENTS, |settings, event| {
        hooks_event(settings, event)
            .and_then(Value::as_array)
            .map(|entries| {
                entries
                    .iter()
                    .flat_map(claude_inner_hooks)
                    .filter(|hook| hook_command_contains(hook, CLAUDE_HOOK_MARKER))
                    .filter_map(hook_command)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default()
    })
}

fn codex_hook_change_summary(before: &Value, after: &Value) -> String {
    hook_change_summary(before, after, &CODEX_HOOK_EVENTS, |settings, event| {
        hooks_event(settings, event)
            .and_then(Value::as_array)
            .map(|entries| {
                entries
                    .iter()
                    .filter(|entry| hook_command_contains(entry, CODEX_HOOK_MARKER))
                    .filter_map(hook_command)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default()
    })
}

fn hooks_event<'a>(settings: &'a Value, event: &str) -> Option<&'a Value> {
    settings
        .get("hooks")
        .and_then(Value::as_object)
        .and_then(|hooks| hooks.get(event))
}

fn hook_change_summary(
    before: &Value,
    after: &Value,
    events: &[&str],
    nabu_commands: impl Fn(&Value, &str) -> Vec<String>,
) -> String {
    let mut added = Vec::new();
    let mut removed = Vec::new();
    let mut replaced = Vec::new();
    for event in events {
        let before_commands = nabu_commands(before, event);
        let after_commands = nabu_commands(after, event);
        match (before_commands.is_empty(), after_commands.is_empty()) {
            (true, false) => added.push(*event),
            (false, true) => removed.push(*event),
            // Stale command swapped for the canonical one (e.g. --home change).
            (false, false) if before_commands != after_commands => replaced.push(*event),
            _ => {}
        }
    }
    let mut lines = Vec::new();
    if !added.is_empty() {
        lines.push(format!("+ nabu hook added to: {}", added.join(", ")));
    }
    if !removed.is_empty() {
        lines.push(format!("- nabu hook removed from: {}", removed.join(", ")));
    }
    if !replaced.is_empty() {
        lines.push(format!("~ nabu hook replaced in: {}", replaced.join(", ")));
    }
    if lines.is_empty() {
        "no nabu hook changes".to_string()
    } else {
        lines.join("\n")
    }
}

fn command_in_path(command: &str) -> bool {
    let Some(paths) = env::var_os("PATH") else {
        return false;
    };
    env::split_paths(&paths).any(|path| path.join(command).is_file())
}

// Refuse valid-but-unexpected JSON rather than coercing it: silently replacing a
// non-object root, non-object `hooks`, or non-array event entry would discard the
// user's data on write.
fn require_object(value: &Value, field: &str) -> Result<()> {
    if value.is_object() {
        Ok(())
    } else {
        Err(Error::Validation(format!(
            "expected `{field}` to be a JSON object, found {}",
            json_type_name(value)
        )))
    }
}

fn require_event_array(value: &Value, event: &str) -> Result<()> {
    if value.is_array() {
        Ok(())
    } else {
        Err(Error::Validation(format!(
            "expected hook event `{event}` to be a JSON array, found {}",
            json_type_name(value)
        )))
    }
}

fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Single-quote a path for safe embedding in a shell command line. A home path
/// containing spaces (or any shell metacharacter) would otherwise word-split and
/// break every hook invocation.
fn shell_single_quote(path: &Path) -> String {
    format!("'{}'", path.to_string_lossy().replace('\'', r"'\''"))
}

/// `create_dir_all` a directory nabu owns, tightening its mode to 0o700 only when
/// this call actually created it — never re-permission a pre-existing user dir.
fn create_dir_all_owned(dir: &Path) -> Result<()> {
    let created = !dir.exists();
    fs::create_dir_all(dir).map_err(|source| Error::Io {
        path: dir.to_path_buf(),
        source,
    })?;
    if created {
        chmod(dir, 0o700)?;
    }
    Ok(())
}

fn write_json_file(path: &Path, value: &Value, mode: u32) -> Result<()> {
    let final_mode = file_mode_or(path, mode)?;
    let content = serde_json::to_vec_pretty(value)?;
    write_atomic(path, &content, final_mode)
}

fn write_text_file(path: &Path, content: &str, mode: u32) -> Result<()> {
    let final_mode = file_mode_or(path, mode)?;
    write_atomic(path, content.as_bytes(), final_mode)
}

/// Write via a temp file in the target's directory, set its mode, then rename over
/// the target: a crash mid-write can never leave the live agent config truncated.
fn write_atomic(path: &Path, content: &[u8], mode: u32) -> Result<()> {
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("config");
    let tmp_path = dir.join(format!(".{file_name}.nabu-tmp.{}", std::process::id()));
    fs::write(&tmp_path, content).map_err(|source| Error::Io {
        path: tmp_path.clone(),
        source,
    })?;
    chmod(&tmp_path, mode)?;
    fs::rename(&tmp_path, path).map_err(|source| {
        let _ = fs::remove_file(&tmp_path);
        Error::Io {
            path: path.to_path_buf(),
            source,
        }
    })
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
        let codex_command = format!(
            "nabu ingest hook --tool codex --home {}",
            shell_single_quote(home)
        );
        let claude_command = format!(
            "nabu ingest hook --tool claude --home {}",
            shell_single_quote(home)
        );
        let rendered_codex: Value = serde_json::from_str(
            &CODEX_HOOKS_REFERENCE.replace("__NABU_HOME__", &home.display().to_string()),
        )
        .unwrap();
        let rendered_claude: Value = serde_json::from_str(
            &CLAUDE_SETTINGS_FRAGMENT_REFERENCE
                .replace("__NABU_HOME__", &home.display().to_string()),
        )
        .unwrap();

        assert_eq!(
            add_codex_hooks(json!({}), &codex_command).unwrap(),
            rendered_codex
        );
        assert_eq!(
            add_claude_hooks(json!({}), &claude_command).unwrap(),
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

    // A claude entry carrying the nabu hook alongside a co-located user hook.
    fn claude_entry_with_user_and_nabu(nabu_command: &str) -> Value {
        json!({
            "hooks": [
                { "type": "command", "command": "echo user-first" },
                { "type": "command", "command": nabu_command }
            ]
        })
    }

    #[test]
    fn install_claude_shell_quotes_home_with_space() {
        let _guard = ENV_LOCK.lock().unwrap();
        let temp = tempdir().unwrap();
        let harness_home = temp.path().join("home with space/raven");
        let claude_config = temp.path().join("claude");
        fs::create_dir_all(&harness_home).unwrap();
        fs::create_dir_all(&claude_config).unwrap();
        let _env = EnvGuard::set([("CLAUDE_CONFIG_DIR", claude_config.as_os_str())]);

        install_claude(&harness_home, false).unwrap();

        let settings: Value =
            serde_json::from_str(&fs::read_to_string(claude_config.join("settings.json")).unwrap())
                .unwrap();
        let command = settings
            .pointer("/hooks/SessionStart/0/hooks/0/command")
            .and_then(Value::as_str)
            .unwrap();
        let expected = format!(
            "nabu ingest hook --tool claude --home {}",
            shell_single_quote(&harness_home)
        );
        assert_eq!(command, expected);
        // The quoted argument must shell-split back to exactly the home path.
        assert!(command.ends_with(&format!("--home '{}'", harness_home.display())));
        assert!(command.contains("home with space"));
    }

    #[test]
    fn add_claude_hooks_refuses_unexpected_json_shapes() {
        // Non-object root.
        assert!(matches!(
            add_claude_hooks(json!([]), "cmd"),
            Err(Error::Validation(_))
        ));
        // `hooks` holds a non-object.
        assert!(matches!(
            add_claude_hooks(json!({ "hooks": "nope" }), "cmd"),
            Err(Error::Validation(_))
        ));
        // An event key holds a non-array.
        assert!(matches!(
            add_claude_hooks(json!({ "hooks": { "Stop": 42 } }), "cmd"),
            Err(Error::Validation(_))
        ));
    }

    #[test]
    fn install_claude_refuses_unexpected_json_without_writing() {
        let _guard = ENV_LOCK.lock().unwrap();
        for raw in [
            "[]",
            "null",
            r#"{"hooks":"not-an-object"}"#,
            r#"{"hooks":{"Stop":"not-an-array"}}"#,
        ] {
            let temp = tempdir().unwrap();
            let claude_config = temp.path().join("claude");
            fs::create_dir_all(&claude_config).unwrap();
            let settings_path = claude_config.join("settings.json");
            fs::write(&settings_path, raw).unwrap();
            let _env = EnvGuard::set([("CLAUDE_CONFIG_DIR", claude_config.as_os_str())]);

            let result = install_claude(temp.path(), false);
            assert!(
                matches!(result, Err(Error::Validation(_))),
                "expected validation error for {raw}"
            );
            assert_eq!(fs::read_to_string(&settings_path).unwrap(), raw);
        }
    }

    #[test]
    fn uninstall_claude_preserves_colocated_user_hook() {
        let command = "nabu ingest hook --tool claude --home /home";
        let before = json!({
            "hooks": {
                "Stop": [ claude_entry_with_user_and_nabu(command) ]
            }
        });
        // Dedupe sees the nabu hook at inner index 1: re-adding leaves one entry.
        assert!(add_claude_hooks(before.clone(), command)
            .unwrap()
            .pointer("/hooks/Stop")
            .and_then(Value::as_array)
            .map(|entries| entries.len() == 1)
            .unwrap());
        // Detection finds the nabu hook even at a non-zero inner index.
        assert!(claude_event_has_nabu_hook(before.pointer("/hooks/Stop")));

        let after = remove_claude_hooks(before);
        // Nabu hook removed, co-located user hook retained, entry not dropped.
        let inner = after
            .pointer("/hooks/Stop/0/hooks")
            .and_then(Value::as_array);
        assert_eq!(inner.map(Vec::len), Some(1));
        assert_eq!(
            after.pointer("/hooks/Stop/0/hooks/0/command"),
            Some(&json!("echo user-first"))
        );
        assert!(!settings_contains_harness_claude_hooks(&after));
    }

    #[test]
    fn uninstall_claude_prunes_only_events_it_emptied() {
        let command = "nabu ingest hook --tool claude --home /home";
        let installed = add_claude_hooks(json!({}), command).unwrap();
        // A user-created empty array on an unmanaged key.
        let mut settings = installed;
        settings["hooks"]["Notification"] = json!([]);

        let after = remove_claude_hooks(settings);
        let hooks = after.get("hooks").and_then(Value::as_object).unwrap();
        // Every managed event key nabu owned is gone.
        for event in CLAUDE_HOOK_EVENTS {
            assert!(!hooks.contains_key(event), "{event} should be pruned");
        }
        // The user's unmanaged empty array survives.
        assert_eq!(hooks.get("Notification"), Some(&json!([])));
    }

    #[test]
    fn uninstall_claude_no_nabu_hooks_reports_unchanged() {
        let _guard = ENV_LOCK.lock().unwrap();
        let temp = tempdir().unwrap();
        let claude_config = temp.path().join("claude");
        fs::create_dir_all(&claude_config).unwrap();
        let settings_path = claude_config.join("settings.json");
        let original = r#"{"theme":"dark","hooks":{"Notification":[]}}"#;
        fs::write(&settings_path, original).unwrap();
        let _env = EnvGuard::set([("CLAUDE_CONFIG_DIR", claude_config.as_os_str())]);

        let report = uninstall_claude(temp.path(), false).unwrap();
        assert!(!report.changed);
        assert_eq!(fs::read_to_string(&settings_path).unwrap(), original);
    }

    // Exactly one nabu hook per claude event, carrying `expected_command`.
    fn assert_single_claude_nabu_hook(settings: &Value, expected_command: &str) {
        for event in CLAUDE_HOOK_EVENTS {
            let nabu_commands: Vec<&str> = settings
                .pointer(&format!("/hooks/{event}"))
                .and_then(Value::as_array)
                .unwrap()
                .iter()
                .flat_map(claude_inner_hooks)
                .filter(|hook| hook_command_contains(hook, CLAUDE_HOOK_MARKER))
                .filter_map(hook_command)
                .collect();
            assert_eq!(nabu_commands, vec![expected_command], "{event}");
        }
    }

    #[test]
    fn reinstall_replaces_stale_unquoted_hook_without_stacking() {
        let _guard = ENV_LOCK.lock().unwrap();
        let temp = tempdir().unwrap();
        let harness_home = temp.path().join("raven");
        let claude_config = temp.path().join("claude");
        fs::create_dir_all(&harness_home).unwrap();
        fs::create_dir_all(&claude_config).unwrap();
        let _env = EnvGuard::set([("CLAUDE_CONFIG_DIR", claude_config.as_os_str())]);

        // Simulate a pre-quoting install: same home, unquoted command.
        let old_command = format!(
            "nabu ingest hook --tool claude --home {}",
            harness_home.display()
        );
        let old_settings = add_claude_hooks(json!({"theme": "dark"}), &old_command).unwrap();
        let settings_path = claude_config.join("settings.json");
        fs::write(
            &settings_path,
            serde_json::to_string_pretty(&old_settings).unwrap(),
        )
        .unwrap();

        let report = install_claude(&harness_home, false).unwrap();
        assert!(report.changed);
        assert!(report.diff.contains("replaced"));

        let after: Value =
            serde_json::from_str(&fs::read_to_string(&settings_path).unwrap()).unwrap();
        assert_eq!(after["theme"], "dark");
        let new_command = format!(
            "nabu ingest hook --tool claude --home {}",
            shell_single_quote(&harness_home)
        );
        assert_single_claude_nabu_hook(&after, &new_command);
    }

    #[test]
    fn reinstall_over_identical_install_is_unchanged() {
        let _guard = ENV_LOCK.lock().unwrap();
        let temp = tempdir().unwrap();
        let harness_home = temp.path().join("raven");
        let claude_config = temp.path().join("claude");
        fs::create_dir_all(&harness_home).unwrap();
        fs::create_dir_all(&claude_config).unwrap();
        let _env = EnvGuard::set([("CLAUDE_CONFIG_DIR", claude_config.as_os_str())]);

        assert!(install_claude(&harness_home, false).unwrap().changed);
        let settings_path = claude_config.join("settings.json");
        let first = fs::read_to_string(&settings_path).unwrap();

        let report = install_claude(&harness_home, false).unwrap();
        assert!(!report.changed);
        assert_eq!(fs::read_to_string(&settings_path).unwrap(), first);
    }

    #[test]
    fn reinstall_after_home_change_converges_to_one_hook() {
        let _guard = ENV_LOCK.lock().unwrap();
        let temp = tempdir().unwrap();
        let old_home = temp.path().join("old-home");
        let new_home = temp.path().join("new-home");
        let claude_config = temp.path().join("claude");
        let codex_home = temp.path().join("codex");
        fs::create_dir_all(&old_home).unwrap();
        fs::create_dir_all(&new_home).unwrap();
        fs::create_dir_all(&claude_config).unwrap();
        fs::create_dir_all(&codex_home).unwrap();
        let _env = EnvGuard::set([
            ("CLAUDE_CONFIG_DIR", claude_config.as_os_str()),
            ("CODEX_HOME", codex_home.as_os_str()),
        ]);

        install_claude(&old_home, false).unwrap();
        install_codex(&old_home, false).unwrap();
        assert!(install_claude(&new_home, false).unwrap().changed);
        assert!(install_codex(&new_home, false).unwrap().changed);

        let claude: Value =
            serde_json::from_str(&fs::read_to_string(claude_config.join("settings.json")).unwrap())
                .unwrap();
        let expected_claude = format!(
            "nabu ingest hook --tool claude --home {}",
            shell_single_quote(&new_home)
        );
        assert_single_claude_nabu_hook(&claude, &expected_claude);

        let codex: Value =
            serde_json::from_str(&fs::read_to_string(codex_home.join("hooks.json")).unwrap())
                .unwrap();
        let expected_codex = format!(
            "nabu ingest hook --tool codex --home {}",
            shell_single_quote(&new_home)
        );
        for event in CODEX_HOOK_EVENTS {
            let nabu_commands: Vec<&str> = codex
                .pointer(&format!("/hooks/{event}"))
                .and_then(Value::as_array)
                .unwrap()
                .iter()
                .filter(|entry| hook_command_contains(entry, CODEX_HOOK_MARKER))
                .filter_map(hook_command)
                .collect();
            assert_eq!(nabu_commands, vec![expected_codex.as_str()], "{event}");
        }
    }

    #[test]
    fn diff_summary_omits_unrelated_user_config() {
        let command = "nabu ingest hook --tool claude --home /home";
        let before = json!({ "env": { "API_KEY": "super-secret" } });
        let after = add_claude_hooks(before.clone(), command).unwrap();
        let summary = claude_hook_change_summary(&before, &after);
        assert!(!summary.contains("super-secret"));
        assert!(!summary.contains("API_KEY"));
        assert!(summary.contains("nabu hook added to"));
        assert!(summary.contains("SessionStart"));
    }

    #[cfg(unix)]
    #[test]
    fn install_claude_preserves_preexisting_dir_mode() {
        let _guard = ENV_LOCK.lock().unwrap();
        let temp = tempdir().unwrap();
        let claude_config = temp.path().join("claude");
        fs::create_dir_all(&claude_config).unwrap();
        set_mode(&claude_config, 0o755);
        let _env = EnvGuard::set([("CLAUDE_CONFIG_DIR", claude_config.as_os_str())]);

        install_claude(temp.path(), false).unwrap();

        // Pre-existing dir keeps its mode; only the file we wrote is tightened.
        assert_eq!(file_mode(&claude_config), 0o755);
        assert_eq!(file_mode(&claude_config.join("settings.json")), 0o600);
    }

    #[cfg(unix)]
    #[test]
    fn install_claude_tightens_created_dir_mode() {
        let _guard = ENV_LOCK.lock().unwrap();
        let temp = tempdir().unwrap();
        let claude_config = temp.path().join("nested/claude");
        let _env = EnvGuard::set([("CLAUDE_CONFIG_DIR", claude_config.as_os_str())]);

        install_claude(temp.path(), false).unwrap();

        assert_eq!(file_mode(&claude_config), 0o700);
    }

    #[test]
    fn write_atomic_leaves_no_temp_file() {
        let temp = tempdir().unwrap();
        let target = temp.path().join("settings.json");
        write_json_file(&target, &json!({ "a": 1 }), 0o600).unwrap();
        write_json_file(&target, &json!({ "a": 2 }), 0o600).unwrap();
        let value: Value = serde_json::from_str(&fs::read_to_string(&target).unwrap()).unwrap();
        assert_eq!(value, json!({ "a": 2 }));
        // No lingering `.settings.json.nabu-tmp.*` sibling.
        let leftovers = fs::read_dir(temp.path())
            .unwrap()
            .filter_map(std::result::Result::ok)
            .filter(|entry| entry.file_name().to_string_lossy().contains(".nabu-tmp."))
            .count();
        assert_eq!(leftovers, 0);
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
