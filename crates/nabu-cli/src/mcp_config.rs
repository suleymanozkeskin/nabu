//! MCP server registration, validation, and probing for the agent tools.
//!
//! Owns `nabu mcp install/uninstall/validate/probe`: the per-tool config
//! mutators (Codex TOML, Claude/OpenCode JSON, OpenCode JSONC via
//! [`crate::jsonc_edit`]), the apply skeleton (read -> rewrite -> diff -> backup
//! -> write -> report), installed-state detection, and the subprocess client
//! probe. Path resolution comes from [`crate::paths::ToolLayout`]; backups and
//! atomic writes from [`crate::backup`].

use crate::backup::{backup_cli_config, read_text_or_empty, text_diff, write_text_config};
use crate::paths::ToolLayout;
use crate::{jsonc_edit, AgentTool, McpConfigAction};
use nabu_adapters::ConfigChangeReport;
use nabu_core::{Error, Tool};
use serde_json::{json, Value};
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Stdio};
use std::time::{Duration, Instant};

pub(crate) fn mcp_apply_all(
    home: &Path,
    tool: AgentTool,
    action: McpConfigAction,
    dry_run: bool,
) -> nabu_core::Result<Vec<ConfigChangeReport>> {
    let tools = selected_agent_tools(tool);
    let mut reports = Vec::with_capacity(tools.len());
    for &tool in tools {
        reports.push(mcp_apply_one(home, tool, action, dry_run)?);
    }
    Ok(reports)
}

fn selected_agent_tools(tool: AgentTool) -> &'static [AgentTool] {
    match tool {
        AgentTool::Codex => &[AgentTool::Codex],
        AgentTool::Claude => &[AgentTool::Claude],
        AgentTool::Opencode => &[AgentTool::Opencode],
        AgentTool::All => &[AgentTool::Codex, AgentTool::Claude, AgentTool::Opencode],
    }
}

pub(crate) fn mcp_apply_one(
    home: &Path,
    tool: AgentTool,
    action: McpConfigAction,
    dry_run: bool,
) -> nabu_core::Result<ConfigChangeReport> {
    match tool {
        AgentTool::Codex => mcp_apply_codex(home, action, dry_run),
        AgentTool::Claude => mcp_apply_claude(home, action, dry_run),
        AgentTool::Opencode => mcp_apply_opencode(home, action, dry_run),
        AgentTool::All => Err(Error::Validation(
            "internal error: all must be expanded before mcp_apply_one".to_string(),
        )),
    }
}

fn mcp_apply_codex(
    home: &Path,
    action: McpConfigAction,
    dry_run: bool,
) -> nabu_core::Result<ConfigChangeReport> {
    let target_path = Tool::Codex.mcp_config_path()?;
    if dry_run {
        let diff = match action {
            McpConfigAction::Install => {
                "planned TOML snippet:\n[mcp_servers.nabu]\ncommand = \"nabu\"\nargs = [\"mcp\", \"serve\", \"--transport\", \"stdio\"]\nenabled = true\n"
            }
            McpConfigAction::Uninstall => {
                "planned removal:\n[mcp_servers.nabu]\n"
            }
        }
        .to_string();
        return Ok(ConfigChangeReport {
            tool: Tool::Codex,
            target_path,
            changed: true,
            dry_run,
            summary: match action {
                McpConfigAction::Install => "dry-run: Codex MCP config plan only",
                McpConfigAction::Uninstall => "dry-run: Codex MCP removal plan only",
            }
            .to_string(),
            diff,
        });
    }
    let before = read_text_or_empty(&target_path)?;
    let after = match action {
        McpConfigAction::Install => add_codex_mcp_block(&before),
        McpConfigAction::Uninstall => {
            let after = remove_toml_table(&before, "[mcp_servers.nabu]");
            // Also strip the pre-rename table so uninstall leaves no orphan.
            remove_toml_table(&after, "[mcp_servers.tupsharrum]")
        }
    };
    let changed = before != after;
    let diff = text_diff(&before, &after);

    if changed && !dry_run {
        backup_cli_config(home, Tool::Codex, mcp_operation_name(action), &target_path)?;
        write_text_config(&target_path, &after, 0o600)?;
    }

    Ok(ConfigChangeReport {
        tool: Tool::Codex,
        target_path,
        changed,
        dry_run,
        summary: match (action, dry_run, changed) {
            (McpConfigAction::Install, true, _) => "dry-run: Codex MCP config diff only",
            (McpConfigAction::Install, false, true) => "installed Codex MCP server config",
            (McpConfigAction::Install, false, false) => "Codex MCP server config already installed",
            (McpConfigAction::Uninstall, true, _) => "dry-run: Codex MCP removal diff only",
            (McpConfigAction::Uninstall, false, true) => "removed Codex MCP server config",
            (McpConfigAction::Uninstall, false, false) => {
                "Codex MCP server config was not installed"
            }
        }
        .to_string(),
        diff,
    })
}

/// Run a `claude` CLI subcommand and capture its output. The claude CLI owns its
/// own MCP server registry, so registration goes through it rather than a file we
/// write ourselves.
fn run_claude_cli(args: &[&str]) -> nabu_core::Result<std::process::Output> {
    ProcessCommand::new("claude")
        .args(args)
        .output()
        .map_err(|source| Error::Io {
            path: PathBuf::from("claude"),
            source,
        })
}

fn mcp_apply_claude(
    home: &Path,
    action: McpConfigAction,
    dry_run: bool,
) -> nabu_core::Result<ConfigChangeReport> {
    let command = match action {
        McpConfigAction::Install => {
            "claude mcp add --scope user --transport stdio nabu -- nabu mcp serve --transport stdio"
        }
        McpConfigAction::Uninstall => "claude mcp remove --scope user nabu",
    };
    let target_path = Tool::Claude.mcp_config_path()?;
    let use_native = command_in_path("claude");

    let before_text = read_text_or_empty(&target_path)?;
    let before: Value = if before_text.trim().is_empty() {
        json!({})
    } else {
        serde_json::from_str(&before_text)?
    };
    let after = match action {
        McpConfigAction::Install => add_claude_mcp(before.clone()),
        McpConfigAction::Uninstall => remove_claude_mcp(before.clone()),
    };
    let after_text = serde_json::to_string_pretty(&after)?;
    let changed = before != after;

    let diff = if dry_run && use_native {
        format!("native command:\n{command}\n")
    } else {
        text_diff(&before_text, &after_text)
    };

    if !dry_run {
        if use_native {
            // The `claude` CLI owns its MCP registry; back up its config, then
            // upsert idempotently. `claude mcp add` errors if the server already
            // exists, and our file-based change detection can disagree with the
            // CLI's own store — so always remove-then-add. Also drop the legacy
            // `tupsharrum` registration so an upgrade leaves no broken entry.
            if target_path.exists() {
                backup_cli_config(home, Tool::Claude, mcp_operation_name(action), &target_path)?;
            }
            let _ = run_claude_cli(&["mcp", "remove", "--scope", "user", "nabu"]);
            let _ = run_claude_cli(&["mcp", "remove", "--scope", "user", "tupsharrum"]);
            if matches!(action, McpConfigAction::Install) {
                let output = run_claude_cli(&[
                    "mcp",
                    "add",
                    "--scope",
                    "user",
                    "--transport",
                    "stdio",
                    "nabu",
                    "--",
                    "nabu",
                    "mcp",
                    "serve",
                    "--transport",
                    "stdio",
                ])?;
                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    let detail = stderr.trim();
                    return Err(Error::Validation(if detail.is_empty() {
                        format!("Claude MCP command exited with status {}", output.status)
                    } else {
                        format!(
                            "Claude MCP command exited with status {}: {detail}",
                            output.status
                        )
                    }));
                }
            }
        } else if changed {
            if target_path.exists() {
                backup_cli_config(home, Tool::Claude, mcp_operation_name(action), &target_path)?;
            }
            write_text_config(&target_path, &after_text, 0o600)?;
        }
    }

    Ok(ConfigChangeReport {
        tool: Tool::Claude,
        target_path,
        changed: if dry_run && use_native { true } else { changed },
        dry_run,
        summary: match (action, dry_run, use_native, changed) {
            (McpConfigAction::Install, true, true, _) => {
                "dry-run: Claude Code MCP native command only"
            }
            (McpConfigAction::Install, true, false, _) => {
                "dry-run: Claude Code MCP user config diff only"
            }
            (McpConfigAction::Install, false, _, false) => {
                "Claude Code MCP config already installed"
            }
            (McpConfigAction::Install, false, true, true) => {
                "installed Claude Code MCP config using native CLI"
            }
            (McpConfigAction::Install, false, false, true) => {
                "installed Claude Code MCP config by writing user config"
            }
            (McpConfigAction::Uninstall, true, true, _) => {
                "dry-run: Claude Code MCP native removal command only"
            }
            (McpConfigAction::Uninstall, true, false, _) => {
                "dry-run: Claude Code MCP user config removal diff only"
            }
            (McpConfigAction::Uninstall, false, _, false) => {
                "Claude Code MCP config was not installed"
            }
            (McpConfigAction::Uninstall, false, true, true) => {
                "removed Claude Code MCP config using native CLI"
            }
            (McpConfigAction::Uninstall, false, false, true) => {
                "removed Claude Code MCP config by writing user config"
            }
        }
        .to_string(),
        diff,
    })
}

pub(crate) fn mcp_apply_opencode(
    home: &Path,
    action: McpConfigAction,
    dry_run: bool,
) -> nabu_core::Result<ConfigChangeReport> {
    let target_path = Tool::Opencode.mcp_config_path()?;
    if dry_run {
        let diff = match action {
            McpConfigAction::Install => {
                "planned JSON entry:\n{\n  \"mcp\": {\n    \"nabu\": {\n      \"type\": \"local\",\n      \"command\": [\"nabu\", \"mcp\", \"serve\", \"--transport\", \"stdio\"],\n      \"enabled\": true\n    }\n  }\n}\n"
            }
            McpConfigAction::Uninstall => "planned removal:\nmcp.nabu\n",
        }
        .to_string();
        return Ok(ConfigChangeReport {
            tool: Tool::Opencode,
            target_path,
            changed: true,
            dry_run,
            summary: match action {
                McpConfigAction::Install => "dry-run: OpenCode MCP config plan only",
                McpConfigAction::Uninstall => "dry-run: OpenCode MCP removal plan only",
            }
            .to_string(),
            diff,
        });
    }
    let before_text = read_text_or_empty(&target_path)?;
    let after_text = rewrite_opencode_mcp_text(&before_text, action)?;
    let changed = before_text != after_text;
    let diff = text_diff(&before_text, &after_text);

    if changed && !dry_run {
        backup_cli_config(
            home,
            Tool::Opencode,
            mcp_operation_name(action),
            &target_path,
        )?;
        write_text_config(&target_path, &after_text, 0o600)?;
    }

    Ok(ConfigChangeReport {
        tool: Tool::Opencode,
        target_path,
        changed,
        dry_run,
        summary: match (action, dry_run, changed) {
            (McpConfigAction::Install, true, _) => "dry-run: OpenCode MCP config diff only",
            (McpConfigAction::Install, false, true) => "installed OpenCode MCP server config",
            (McpConfigAction::Install, false, false) => {
                "OpenCode MCP server config already installed"
            }
            (McpConfigAction::Uninstall, true, _) => "dry-run: OpenCode MCP removal diff only",
            (McpConfigAction::Uninstall, false, true) => "removed OpenCode MCP server config",
            (McpConfigAction::Uninstall, false, false) => {
                "OpenCode MCP server config was not installed"
            }
        }
        .to_string(),
        diff,
    })
}

pub(crate) fn mcp_validate_all(home: &Path, tool: AgentTool) -> nabu_core::Result<Value> {
    let mut value = json!({});
    let fixture = mcp_fixture_validation(home)?;
    for &tool in selected_agent_tools(tool) {
        match tool {
            AgentTool::Codex => {
                let client = mcp_client_probe("codex", &["mcp", "get", "nabu", "--json"])?;
                let installed = command_in_path("codex");
                let entry_installed = codex_mcp_entry_installed();
                value["codex"] = json!({
                    "status": mcp_validation_status(installed, entry_installed, &client, &fixture),
                    "client_installed": installed,
                    "mcp_entry_installed": entry_installed,
                    "client_list": client,
                    "fixture_server": fixture.clone(),
                    "search_history_advertised": fixture["search_history_advertised"]
                });
            }
            AgentTool::Claude => {
                let client = mcp_client_probe("claude", &["mcp", "get", "nabu"])?;
                let installed = command_in_path("claude");
                let entry_installed = claude_mcp_entry_installed();
                value["claude"] = json!({
                    "status": mcp_validation_status(installed, entry_installed, &client, &fixture),
                    "client_installed": installed,
                    "mcp_entry_installed": entry_installed,
                    "client_list": client,
                    "fixture_server": fixture.clone(),
                    "search_history_advertised": fixture["search_history_advertised"]
                });
            }
            AgentTool::Opencode => {
                let client = mcp_client_probe("opencode", &["mcp", "list", "--pure"])?;
                let installed = command_in_path("opencode");
                let entry_installed = opencode_mcp_entry_installed();
                value["opencode"] = json!({
                    "status": mcp_validation_status(installed, entry_installed, &client, &fixture),
                    "client_installed": installed,
                    "mcp_entry_installed": entry_installed,
                    "client_list": client,
                    "fixture_server": fixture.clone(),
                    "search_history_advertised": fixture["search_history_advertised"]
                });
            }
            AgentTool::All => {}
        }
    }
    Ok(value)
}

fn mcp_validation_status(
    client_installed: bool,
    entry_installed: bool,
    client: &Value,
    fixture: &Value,
) -> &'static str {
    if !client_installed {
        return "not_applicable";
    }
    if !entry_installed {
        return "not_configured";
    }
    if client.get("status").and_then(Value::as_str) != Some("ok") {
        return "upstream_unhealthy";
    }
    if client.get("contains_nabu").and_then(Value::as_bool) != Some(true) {
        return "upstream_unhealthy";
    }
    if fixture
        .get("search_history_advertised")
        .and_then(Value::as_bool)
        != Some(true)
        || fixture.get("fixture_query_ok").and_then(Value::as_bool) != Some(true)
    {
        return "server_unhealthy";
    }
    "ok"
}

fn mcp_fixture_validation(home: &Path) -> nabu_core::Result<Value> {
    let fixture_home = mcp_fixture_home(home);
    let mut input = Vec::new();
    for message in [
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
        json!({"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}),
        json!({
            "jsonrpc":"2.0",
            "id":3,
            "method":"tools/call",
            "params":{
                "name":"search_history",
                "arguments":{"query":"fixture marker","limit":1}
            }
        }),
    ] {
        serde_json::to_writer(&mut input, &message)?;
        input.push(b'\n');
    }
    let mut output = Vec::new();
    nabu_mcp::serve_with_io(fixture_home.clone(), Cursor::new(input), &mut output)?;
    let mut initialize_ok = false;
    let mut search_history_advertised = false;
    let mut fixture_query_ok = false;
    for line in String::from_utf8_lossy(&output).lines() {
        let Ok(response) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        match response.get("id").and_then(Value::as_i64) {
            Some(1) => {
                initialize_ok |= response
                    .pointer("/result/serverInfo/name")
                    .and_then(Value::as_str)
                    == Some("nabu");
            }
            Some(2) => {
                search_history_advertised |= response
                    .pointer("/result/tools")
                    .and_then(Value::as_array)
                    .map(|tools| {
                        tools.iter().any(|tool| {
                            tool.get("name").and_then(Value::as_str) == Some("search_history")
                        })
                    })
                    .unwrap_or(false);
            }
            Some(3) => {
                fixture_query_ok |= response.pointer("/result/isError").and_then(Value::as_bool)
                    == Some(false)
                    && response
                        .pointer("/result/structuredContent/results")
                        .and_then(Value::as_array)
                        .map(|results| !results.is_empty())
                        .unwrap_or(false);
            }
            _ => {}
        }
    }

    Ok(json!({
        "fixture_home": fixture_home,
        "initialize_ok": initialize_ok,
        "search_history_advertised": search_history_advertised,
        "fixture_query_ok": fixture_query_ok
    }))
}

fn mcp_fixture_home(home: &Path) -> PathBuf {
    let fixture_home = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("fixtures")
        .join("acceptance-home");
    if fixture_home.join("index").join("harness.db").is_file() {
        fixture_home.canonicalize().unwrap_or(fixture_home)
    } else {
        home.to_path_buf()
    }
}

fn mcp_client_probe(command: &str, args: &[&str]) -> nabu_core::Result<Value> {
    if !command_in_path(command) {
        return Ok(json!({
            "attempted": false,
            "status": "not_applicable",
            "command": command,
            "args": args,
            "contains_nabu": false
        }));
    }
    let result = run_command_capture(command, args, Duration::from_secs(8))?;
    let command_ok = !result.timed_out && result.exit_code == Some(0);
    Ok(json!({
        "attempted": true,
        "command": command,
        "args": args,
        "status": if result.timed_out {
            "timeout"
        } else if result.exit_code == Some(0) {
            "ok"
        } else {
            "error"
        },
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "contains_nabu": command_ok && (result.stdout.contains("nabu") || result.stderr.contains("nabu")),
        "stdout": bounded_probe_text(&result.stdout),
        "stderr": bounded_probe_text(&result.stderr)
    }))
}

struct CommandCapture {
    exit_code: Option<i32>,
    timed_out: bool,
    stdout: String,
    stderr: String,
}

fn run_command_capture(
    command: &str,
    args: &[&str],
    timeout: Duration,
) -> nabu_core::Result<CommandCapture> {
    let mut child = ProcessCommand::new(command)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|source| Error::Io {
            path: PathBuf::from(command),
            source,
        })?;
    let started = Instant::now();
    loop {
        if let Some(status) = child.try_wait().map_err(|source| Error::Io {
            path: PathBuf::from(command),
            source,
        })? {
            let output = child.wait_with_output().map_err(|source| Error::Io {
                path: PathBuf::from(command),
                source,
            })?;
            return Ok(CommandCapture {
                exit_code: status.code(),
                timed_out: false,
                stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            });
        }
        if started.elapsed() >= timeout {
            let _ = child.kill();
            let output = child.wait_with_output().map_err(|source| Error::Io {
                path: PathBuf::from(command),
                source,
            })?;
            return Ok(CommandCapture {
                exit_code: None,
                timed_out: true,
                stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            });
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn bounded_probe_text(value: &str) -> String {
    const MAX_PROBE_CHARS: usize = 4 * 1024;
    match value.char_indices().nth(MAX_PROBE_CHARS) {
        Some((truncate_at, _)) => {
            let mut truncated = String::with_capacity(truncate_at + 3);
            truncated.push_str(&value[..truncate_at]);
            truncated.push_str("...");
            truncated
        }
        None => value.to_string(),
    }
}

pub(crate) fn codex_mcp_entry_installed() -> bool {
    Tool::Codex
        .mcp_config_path()
        .ok()
        .and_then(|path| read_text_or_empty(&path).ok())
        .map(|content| content.contains("[mcp_servers.nabu]"))
        .unwrap_or(false)
}

pub(crate) fn claude_mcp_entry_installed() -> bool {
    Tool::Claude
        .mcp_config_path()
        .ok()
        .and_then(|path| read_text_or_empty(&path).ok())
        .and_then(|content| serde_json::from_str::<Value>(&content).ok())
        .and_then(|config| config.pointer("/mcpServers/nabu").cloned())
        .is_some()
}

pub(crate) fn opencode_mcp_entry_installed() -> bool {
    Tool::Opencode
        .mcp_config_path()
        .ok()
        .and_then(|path| read_text_or_empty(&path).ok())
        .map(|content| {
            serde_json::from_str::<Value>(&content)
                .ok()
                .and_then(|config| config.pointer("/mcp/nabu").cloned())
                .is_some()
                || jsonc_edit::opencode_mcp_text_entry_installed(&content)
        })
        .unwrap_or(false)
}

pub(crate) fn add_codex_mcp_block(content: &str) -> String {
    let mut output = remove_toml_table(content, "[mcp_servers.nabu]");
    // Drop the pre-rename table too, so a re-install leaves no orphaned server.
    output = remove_toml_table(&output, "[mcp_servers.tupsharrum]");
    let trimmed_len = output.trim_end_matches('\n').len();
    output.truncate(trimmed_len);
    if !output.is_empty() {
        output.push_str("\n\n");
    }
    output.push_str(
        "[mcp_servers.nabu]\ncommand = \"nabu\"\nargs = [\"mcp\", \"serve\", \"--transport\", \"stdio\"]\nenabled = true\n",
    );
    output
}

pub(crate) fn add_claude_mcp(mut config: Value) -> Value {
    ensure_object(&mut config);
    let object = config.as_object_mut().expect("config object");
    let mcp_servers = object.entry("mcpServers").or_insert_with(|| json!({}));
    ensure_object(mcp_servers);
    let servers = mcp_servers.as_object_mut().expect("mcpServers object");
    // No orphans: drop the pre-rename server while installing the current one.
    servers.remove("tupsharrum");
    servers.insert(
        "nabu".to_string(),
        json!({
            "type": "stdio",
            "command": "nabu",
            "args": ["mcp", "serve", "--transport", "stdio"]
        }),
    );
    config
}

fn remove_claude_mcp(mut config: Value) -> Value {
    if let Some(mcp_servers) = config.get_mut("mcpServers").and_then(Value::as_object_mut) {
        mcp_servers.remove("nabu");
        // Also remove the pre-rename key so an upgrade leaves no orphaned server.
        mcp_servers.remove("tupsharrum");
        if mcp_servers.is_empty() {
            config
                .as_object_mut()
                .expect("config object")
                .remove("mcpServers");
        }
    }
    config
}

fn remove_toml_table(content: &str, table_header: &str) -> String {
    let mut output = String::with_capacity(content.len());
    let mut skipping = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed == table_header {
            skipping = true;
            continue;
        }
        if skipping && trimmed.starts_with('[') && trimmed.ends_with(']') {
            skipping = false;
        }
        if !skipping {
            if !output.is_empty() {
                output.push('\n');
            }
            output.push_str(line);
        }
    }
    if content.ends_with('\n') && !output.is_empty() {
        output.push('\n');
    }
    output
}

fn rewrite_opencode_mcp_text(content: &str, action: McpConfigAction) -> nabu_core::Result<String> {
    if content.trim().is_empty() {
        let value = match action {
            McpConfigAction::Install => add_opencode_mcp(json!({})),
            McpConfigAction::Uninstall => json!({}),
        };
        return Ok(serde_json::to_string_pretty(&value)?);
    }

    if let Some(rewritten) =
        jsonc_edit::rewrite_opencode_mcp_text_preserving_layout(content, action)?
    {
        return Ok(rewritten);
    }

    let before: Value = serde_json::from_str(content)?;
    let after = match action {
        McpConfigAction::Install => add_opencode_mcp(before),
        McpConfigAction::Uninstall => remove_opencode_mcp(before),
    };
    Ok(serde_json::to_string_pretty(&after)?)
}

pub(crate) fn add_opencode_mcp(mut config: Value) -> Value {
    ensure_object(&mut config);
    let object = config.as_object_mut().expect("config object");
    let mcp = object.entry("mcp").or_insert_with(|| json!({}));
    ensure_object(mcp);
    let mcp_obj = mcp.as_object_mut().expect("mcp object");
    // No orphans: drop the pre-rename server while installing the current one, so
    // an upgrade (or re-install) leaves no entry pointing at a removed binary.
    mcp_obj.remove("tupsharrum");
    mcp_obj.insert(
        "nabu".to_string(),
        json!({
            "type": "local",
            "command": ["nabu", "mcp", "serve", "--transport", "stdio"],
            "enabled": true
        }),
    );
    config
}

fn remove_opencode_mcp(mut config: Value) -> Value {
    if let Some(mcp) = config.get_mut("mcp").and_then(Value::as_object_mut) {
        mcp.remove("nabu");
        // Also remove the pre-rename key so an upgrade leaves no orphaned server.
        mcp.remove("tupsharrum");
        if mcp.is_empty() {
            config.as_object_mut().expect("config object").remove("mcp");
        }
    }
    config
}

fn ensure_object(value: &mut Value) {
    if !value.is_object() {
        *value = json!({});
    }
}

fn mcp_operation_name(action: McpConfigAction) -> &'static str {
    match action {
        McpConfigAction::Install => "mcp-install",
        McpConfigAction::Uninstall => "mcp-uninstall",
    }
}

fn command_in_path(command: &str) -> bool {
    let Some(paths) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&paths).any(|mut path| {
        path.push(command);
        path.is_file()
    })
}
