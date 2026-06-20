//! Output rendering: JSON envelopes, CLI error frames, doctor reports, and the
//! Display newtypes used by human-readable output.
//!
//! Splits the machine-readable surface (`json_success`/`json_error` and the
//! `doctor_json_data` builder) from the human surface (the `print_*` helpers and
//! the `ByteSize`/`AlsoAt`/`OptionalValue`/`OptionalField` formatting newtypes),
//! so both can be asserted in tests independently of command dispatch.

use crate::mcp_config::{
    claude_mcp_entry_installed, codex_mcp_entry_installed, opencode_mcp_entry_installed,
};
use crate::{DoctorTool, OutputFormat};
use nabu_adapters::{claude_status, codex_status, opencode_status, ConfigChangeReport};
use nabu_core::{
    doctor_with_options, latest_event, Corroboration, Error, PurgeAction, PurgeAllReport,
    PurgeTier, SearchPage, SessionPage, Tool,
};
use serde_json::{json, Value};
use std::path::Path;

pub(crate) fn doctor_json_data(
    home: &Path,
    tool: DoctorTool,
    deep: bool,
) -> nabu_core::Result<Value> {
    let report = doctor_with_options(home, deep);
    let mut value = serde_json::to_value(&report)?;
    if matches!(tool, DoctorTool::Claude | DoctorTool::All) {
        let claude = claude_status(home)?;
        value["tools"]["claude"] = json!({
            "status": tool_status_label(claude.claude_installed, claude.hooks_installed),
            "settings_path": claude.settings_path,
            "hooks_installed": claude.hooks_installed,
            "mcp_entry_installed": claude_mcp_entry_installed(),
            "claude_installed": claude.claude_installed,
            "trusted_active": null,
            "storage_writable": claude.storage_writable,
            "latest_captured_event": latest_event(home, Tool::Claude)?
        });
    }
    if matches!(tool, DoctorTool::Codex | DoctorTool::All) {
        let codex = codex_status(home)?;
        value["tools"]["codex"] = json!({
            "status": tool_status_label(codex.codex_installed, codex.hooks_installed),
            "hooks_path": codex.hooks_path,
            "hooks_installed": codex.hooks_installed,
            "mcp_entry_installed": codex_mcp_entry_installed(),
            "codex_installed": codex.codex_installed,
            "trust_guidance": codex.trust_guidance,
            "storage_writable": codex.storage_writable,
            "latest_captured_event": latest_event(home, Tool::Codex)?
        });
    }
    if matches!(tool, DoctorTool::Opencode | DoctorTool::All) {
        let opencode = opencode_status(home)?;
        value["tools"]["opencode"] = json!({
            "status": tool_status_label(opencode.opencode_installed, opencode.plugin_installed),
            "plugin_path": opencode.plugin_path,
            "plugin_installed": opencode.plugin_installed,
            "mcp_entry_installed": opencode_mcp_entry_installed(),
            "config_status": opencode.config_status,
            "server_url": opencode.server_url,
            "reconciliation_enabled": opencode.reconciliation_enabled,
            "opencode_installed": opencode.opencode_installed,
            "storage_writable": opencode.storage_writable,
            "latest_captured_event": latest_event(home, Tool::Opencode)?
        });
    }
    Ok(value)
}

fn tool_status_label(upstream_installed: bool, adapter_installed: bool) -> &'static str {
    if !upstream_installed {
        "not_applicable"
    } else if !adapter_installed {
        "not_configured"
    } else {
        "ok"
    }
}

pub(crate) fn json_success(data: Value) -> Value {
    json!({
        "ok": true,
        "data": data
    })
}

pub(crate) fn json_error(error: &Error) -> Value {
    let code = cli_error_code(error);
    json!({
        "ok": false,
        "error": {
            "code": code,
            "message": error.to_string(),
            "recoverable": code != "INTERNAL_ERROR",
            "hint": cli_error_hint(code),
            "details": cli_error_details(error)
        }
    })
}

fn cli_error_code(error: &Error) -> &'static str {
    match error {
        Error::Validation(message) if cli_validation_message_is_not_found(message) => "NOT_FOUND",
        Error::SemanticUnavailable(_) => "SEMANTIC_UNAVAILABLE",
        Error::Validation(_) | Error::Json(_) => "VALIDATION_ERROR",
        Error::Io { source, .. } if source.kind() == std::io::ErrorKind::PermissionDenied => {
            "PERMISSION_DENIED"
        }
        Error::HomeUnavailable | Error::Io { .. } => "STORAGE_UNAVAILABLE",
        Error::Sqlite { .. } => "INDEX_UNAVAILABLE",
        Error::TimeFormat(_) => "INTERNAL_ERROR",
    }
}

fn cli_validation_message_is_not_found(message: &str) -> bool {
    message.starts_with("session not found for ")
        || message.starts_with("event not found for ")
        || (message.starts_with("raw line ") && message.split_once(" not found in ").is_some())
}

fn cli_error_hint(code: &str) -> &'static str {
    match code {
        "VALIDATION_ERROR" => "Fix the command arguments or input JSON and retry.",
        "NOT_FOUND" => "Run a discovery command such as search or show with a known session.",
        "PERMISSION_DENIED" => "Check ownership and filesystem permissions for the reported path.",
        "STORAGE_UNAVAILABLE" => "Run nabu init and check filesystem permissions.",
        "INDEX_UNAVAILABLE" => "Run nabu index --once and retry.",
        "SEMANTIC_UNAVAILABLE" => {
            "Use --mode lexical, install the model with embed download --yes, or rebuild with --features semantic."
        }
        _ => "Retry with human output and file an issue if it persists.",
    }
}

fn cli_error_details(error: &Error) -> Value {
    match error {
        Error::Io { path, source } if source.kind() == std::io::ErrorKind::PermissionDenied => {
            json!({
                "path": path.display().to_string(),
                "attempted_operation": "filesystem access"
            })
        }
        _ => json!({}),
    }
}

pub(crate) fn cli_exit_code(error: &Error) -> i32 {
    match cli_error_code(error) {
        "VALIDATION_ERROR" | "NOT_FOUND" => 1,
        "STORAGE_UNAVAILABLE" | "INDEX_UNAVAILABLE" | "PERMISSION_DENIED" => 3,
        _ => 5,
    }
}

pub(crate) struct ByteSize(pub(crate) u64);

impl std::fmt::Display for ByteSize {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
        let bytes = self.0;
        let mut value = bytes as f64;
        let mut unit = 0;
        while value >= 1024.0 && unit < UNITS.len() - 1 {
            value /= 1024.0;
            unit += 1;
        }
        if unit == 0 {
            write!(formatter, "{bytes} {}", UNITS[unit])
        } else {
            write!(formatter, "{value:.1} {}", UNITS[unit])
        }
    }
}

pub(crate) fn print_purge_all_preview(preview: &PurgeAllReport, hooks: &[ConfigChangeReport]) {
    println!("nabu purge --all");
    println!("home: {}", preview.home.display());
    println!();
    println!("integrations (hooks live in each tool's own config, not in the store):");
    for hook in hooks {
        let state = if hook.changed {
            "will remove"
        } else {
            "not installed"
        };
        println!(
            "  {:<9} {:<13} {}",
            hook.tool,
            state,
            hook.target_path.display()
        );
    }
    println!();
    println!("store artifacts under the home:");
    for artifact in &preview.artifacts {
        let note = match artifact.action {
            PurgeAction::Absent => "absent",
            PurgeAction::Preserved => "[preserved]",
            PurgeAction::WouldRemove | PurgeAction::Removed => match artifact.tier {
                PurgeTier::Authoritative => "remove  ⚠ IRREVERSIBLE",
                PurgeTier::Derived => "remove  (rebuildable from raw)",
                PurgeTier::Model | PurgeTier::Config => "remove",
            },
        };
        println!(
            "  {:<13} {:>10}   {}",
            artifact.name,
            ByteSize(artifact.bytes),
            note
        );
    }
    println!();
    if preview.unknown_entries.is_empty() {
        println!("untouched (non-nabu files): none");
    } else {
        println!("untouched (non-nabu files, left in place):");
        for path in &preview.unknown_entries {
            println!("  {}", path.display());
        }
    }
    println!();
    println!("total to remove: {}", ByteSize(preview.bytes_in_scope));
    if preview.authoritative_in_scope {
        println!(
            "⚠ includes raw/ — sessions no longer held by the native tool stores cannot be recovered."
        );
    }
    println!("note: the installed `nabu` binary is not removed (it lives outside the store).");
}

pub(crate) fn print_purge_all_result(report: &PurgeAllReport) {
    println!("\nremoved:");
    for artifact in &report.artifacts {
        if artifact.action == PurgeAction::Removed {
            println!("  {:<13} {:>10}", artifact.name, ByteSize(artifact.bytes));
        }
    }
    for artifact in &report.artifacts {
        if artifact.action == PurgeAction::Preserved {
            println!("  {:<13} kept", artifact.name);
        }
    }
    println!("\nfreed {}.", ByteSize(report.bytes_reclaimed));
    if !report.unknown_entries.is_empty() {
        println!(
            "left {} non-nabu entr{} untouched under {}.",
            report.unknown_entries.len(),
            if report.unknown_entries.len() == 1 {
                "y"
            } else {
                "ies"
            },
            report.home.display()
        );
    }
    println!("nabu artifacts removed. Reinstall any time with `nabu wizard` or `nabu install`.");
}

pub(crate) fn print_tool_doctor_human(home: &Path, tool: DoctorTool) -> nabu_core::Result<()> {
    if matches!(tool, DoctorTool::Claude | DoctorTool::All) {
        let status = claude_status(home)?;
        println!("claude.installed={}", status.claude_installed);
        println!("claude.hooks_installed={}", status.hooks_installed);
        println!("claude.storage_writable={}", status.storage_writable);
        println!("claude.settings_path={}", status.settings_path.display());
    }
    if matches!(tool, DoctorTool::Codex | DoctorTool::All) {
        let status = codex_status(home)?;
        println!("codex.installed={}", status.codex_installed);
        println!("codex.hooks_installed={}", status.hooks_installed);
        println!("codex.storage_writable={}", status.storage_writable);
        println!("codex.hooks_path={}", status.hooks_path.display());
        println!("codex.trust_guidance={}", status.trust_guidance);
    }
    if matches!(tool, DoctorTool::Opencode | DoctorTool::All) {
        let status = opencode_status(home)?;
        println!("opencode.installed={}", status.opencode_installed);
        println!("opencode.plugin_installed={}", status.plugin_installed);
        println!("opencode.storage_writable={}", status.storage_writable);
        println!("opencode.plugin_path={}", status.plugin_path.display());
        println!("opencode.config_status={}", status.config_status);
        println!(
            "opencode.reconciliation_enabled={}",
            status.reconciliation_enabled
        );
        if let Some(server_url) = status.server_url {
            println!("opencode.server_url={server_url}");
        }
    }
    Ok(())
}

pub(crate) struct AlsoAt<'a>(pub(crate) &'a [i64]);

impl std::fmt::Display for AlsoAt<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0.is_empty() {
            return Ok(());
        }
        formatter.write_str(" also_at=[")?;
        for (index, raw_line) in self.0.iter().enumerate() {
            if index > 0 {
                formatter.write_str(",")?;
            }
            write!(formatter, "{raw_line}")?;
        }
        formatter.write_str("]")
    }
}

pub(crate) struct OptionalValue<T>(pub(crate) Option<T>);

impl<T: std::fmt::Display> std::fmt::Display for OptionalValue<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            Some(value) => write!(formatter, "{value}"),
            None => formatter.write_str("none"),
        }
    }
}

struct OptionalField<'a> {
    label: &'static str,
    value: Option<&'a str>,
    markdown: bool,
}

impl<'a> OptionalField<'a> {
    fn plain(label: &'static str, value: Option<&'a str>) -> Self {
        Self {
            label,
            value,
            markdown: false,
        }
    }

    fn markdown(label: &'static str, value: Option<&'a str>) -> Self {
        Self {
            label,
            value,
            markdown: true,
        }
    }
}

impl std::fmt::Display for OptionalField<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Some(value) = self.value else {
            return Ok(());
        };
        if self.markdown {
            write!(formatter, " {}=`{}`", self.label, value)
        } else {
            write!(formatter, " {}={}", self.label, value)
        }
    }
}

pub(crate) fn print_corroboration_human(corroboration: Option<&Corroboration>) {
    let Some(corroboration) = corroboration else {
        return;
    };
    if corroboration.refs.is_empty() {
        return;
    }
    println!(
        "  corroboration repo={}",
        corroboration.repo.as_deref().unwrap_or("none")
    );
    for reference in &corroboration.refs {
        println!(
            "    {} {} {}{}{}",
            reference.kind,
            reference.reference,
            reference.status,
            OptionalField::plain("detail", reference.detail.as_deref()),
            OptionalField::plain("reason", reference.reason.as_deref())
        );
    }
}

pub(crate) fn print_corroboration_markdown(corroboration: Option<&Corroboration>) {
    let Some(corroboration) = corroboration else {
        return;
    };
    if corroboration.refs.is_empty() {
        return;
    }
    println!(
        "\n  corroboration repo: `{}`",
        corroboration.repo.as_deref().unwrap_or("none")
    );
    for reference in &corroboration.refs {
        println!(
            "  - `{}` `{}` `{}`{}{}",
            reference.kind,
            reference.reference,
            reference.status,
            OptionalField::markdown("detail", reference.detail.as_deref()),
            OptionalField::markdown("reason", reference.reason.as_deref())
        );
    }
}

/// The summary/target/diff block printed for one applied config change. Returned
/// as a string so it can be asserted without capturing stdout; byte-identical to
/// the three `println!` calls it replaces (each line plus a trailing newline).
pub(crate) fn render_config_change(report: &ConfigChangeReport) -> String {
    format!(
        "{}\ntarget: {}\n{}\n",
        report.summary,
        report.target_path.display(),
        report.diff
    )
}

pub(crate) fn print_config_change(report: &ConfigChangeReport) {
    print!("{}", render_config_change(report));
}

pub(crate) fn print_search_page(page: SearchPage, format: OutputFormat) -> nabu_core::Result<()> {
    match format {
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&page)?);
        }
        OutputFormat::Markdown => {
            println!(
                "returned: {}  truncated: {}  next_offset: {}  max_snippet_chars_applied: {}",
                page.returned,
                page.truncated,
                OptionalValue(
                    page.continuation
                        .as_ref()
                        .map(|continuation| continuation.next_offset)
                ),
                page.max_snippet_chars_applied
            );
            for result in page.results {
                println!(
                    "- `{}` `{}` `{}` score={:.3} `{}` {}:{}{}\n  {}",
                    result.tool,
                    result.session_id,
                    result.canonical_type,
                    result.score,
                    result.timestamp,
                    result.raw_file,
                    result.raw_line,
                    AlsoAt(&result.also_at),
                    result.snippet
                );
                print_corroboration_markdown(result.corroboration.as_ref());
            }
        }
        OutputFormat::Human => {
            println!(
                "returned={} truncated={} next_offset={} max_snippet_chars_applied={}",
                page.returned,
                page.truncated,
                OptionalValue(
                    page.continuation
                        .as_ref()
                        .map(|continuation| continuation.next_offset)
                ),
                page.max_snippet_chars_applied
            );
            for result in page.results {
                println!(
                    "{} {} {}:{} score={:.3}{} {}",
                    result.tool,
                    result.session_id,
                    result.raw_file,
                    result.raw_line,
                    result.score,
                    AlsoAt(&result.also_at),
                    result.snippet
                );
                print_corroboration_human(result.corroboration.as_ref());
            }
        }
    }

    Ok(())
}

pub(crate) fn print_session_page(page: SessionPage, format: OutputFormat) -> nabu_core::Result<()> {
    match format {
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&page)?);
        }
        OutputFormat::Markdown => {
            println!(
                "mode: {}  truncated: {}  next_after_raw_line: {}",
                page.mode,
                page.truncated,
                OptionalValue(page.next_after_raw_line)
            );
            for event in page.events {
                println!(
                    "## {} {}:{}\n\n{}\n",
                    event.canonical_type, event.raw_file, event.raw_line, event.text
                );
                print_corroboration_markdown(event.corroboration.as_ref());
            }
        }
        OutputFormat::Human => {
            println!(
                "mode={} truncated={} next_after_raw_line={}",
                page.mode,
                page.truncated,
                OptionalValue(page.next_after_raw_line)
            );
            for event in page.events {
                println!(
                    "{} {}:{} {}",
                    event.canonical_type, event.raw_file, event.raw_line, event.text
                );
                print_corroboration_human(event.corroboration.as_ref());
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn config_change_render_matches_the_three_line_block() {
        let report = ConfigChangeReport {
            tool: Tool::Codex,
            target_path: PathBuf::from("/tmp/config.toml"),
            changed: true,
            dry_run: false,
            summary: "installed Codex MCP server config".to_string(),
            diff: "--- before\n--- after".to_string(),
        };
        assert_eq!(
            render_config_change(&report),
            "installed Codex MCP server config\ntarget: /tmp/config.toml\n--- before\n--- after\n"
        );
    }
}
