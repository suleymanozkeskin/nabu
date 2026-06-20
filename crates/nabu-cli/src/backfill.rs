//! Backfill: discover and ingest existing tool transcripts.
//!
//! Backs `nabu backfill` (and its dry-run). For each selected tool it visits the
//! transcript roots from [`crate::paths::ToolLayout`] and hands them to the
//! nabu_core backfill engine. OpenCode additionally exposes sessions over a
//! local HTTP API; `backfill_opencode_server_api_if_configured` reconciles those
//! concurrently via [`crate::opencode_http`].

use crate::paths::ToolLayout;
use crate::progress::ProgressEmitter;
use crate::{opencode_http, BackfillTool};
use nabu_core::{
    backfill_dry_run_with_progress, backfill_since_with_progress, ingest_opencode_server_messages,
    opencode_server_url, BackfillReport, Error, Tool,
};
use serde_json::Value;
use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::BufReader;
use std::path::Path;

/// Number of OpenCode sessions reconciled concurrently over the server HTTP API.
const OPENCODE_RECONCILE_FETCH_CONCURRENCY: usize = 8;

pub(crate) fn run_backfill_command(
    home: &Path,
    tool: BackfillTool,
    path: Option<&Path>,
    since: Option<&str>,
    progress: ProgressEmitter,
) -> nabu_core::Result<BackfillReport> {
    if let Some(path) = path {
        return backfill_since_with_progress(home, tool.selection(), path, since, |event| {
            progress.emit_backfill(event)
        });
    }

    let mut report = empty_backfill_report();
    visit_default_backfill_roots(tool, |tool, root| {
        merge_backfill_report(
            &mut report,
            backfill_since_with_progress(home, Some(tool), root, since, |event| {
                progress.emit_backfill(event)
            })?,
        );
        if tool == Tool::Opencode {
            merge_backfill_report(
                &mut report,
                backfill_opencode_server_api_if_configured(home, root, since)?,
            );
        }
        Ok(())
    })?;
    Ok(report)
}

fn empty_backfill_report() -> BackfillReport {
    BackfillReport {
        source_files: 0,
        appended_events: 0,
        checkpoint_files: 0,
        discontinuities: 0,
    }
}

fn merge_backfill_report(target: &mut BackfillReport, source: BackfillReport) {
    target.source_files += source.source_files;
    target.appended_events += source.appended_events;
    target.checkpoint_files += source.checkpoint_files;
    target.discontinuities += source.discontinuities;
}

pub(crate) fn run_backfill_dry_run_command(
    home: &Path,
    tool: BackfillTool,
    path: Option<&Path>,
    since: Option<&str>,
    progress: ProgressEmitter,
) -> nabu_core::Result<nabu_core::BackfillDryRunReport> {
    if let Some(path) = path {
        return backfill_dry_run_with_progress(home, tool.selection(), path, since, |event| {
            progress.emit_backfill(event)
        });
    }

    let mut merged = nabu_core::BackfillDryRunReport {
        source_files: 0,
        on_disk_events: 0,
        captured_events: 0,
        missing_events: 0,
        partial_sessions: 0,
        sessions: Vec::new(),
    };
    visit_default_backfill_roots(tool, |tool, root| {
        let report = backfill_dry_run_with_progress(home, Some(tool), root, since, |event| {
            progress.emit_backfill(event)
        })?;
        merged.source_files += report.source_files;
        merged.on_disk_events += report.on_disk_events;
        merged.captured_events += report.captured_events;
        merged.missing_events += report.missing_events;
        merged.partial_sessions += report.partial_sessions;
        merged.sessions.extend(report.sessions);
        Ok(())
    })?;
    Ok(merged)
}

fn visit_default_backfill_roots(
    selection: BackfillTool,
    mut visit: impl FnMut(Tool, &Path) -> nabu_core::Result<()>,
) -> nabu_core::Result<()> {
    for &tool in selected_backfill_tools(selection) {
        for root in tool.transcript_roots()? {
            visit(tool, &root)?;
        }
    }
    Ok(())
}

fn selected_backfill_tools(selection: BackfillTool) -> &'static [Tool] {
    match selection {
        BackfillTool::Codex => &[Tool::Codex],
        BackfillTool::Claude => &[Tool::Claude],
        BackfillTool::Opencode => &[Tool::Opencode],
        BackfillTool::All => &[Tool::Codex, Tool::Claude, Tool::Opencode],
    }
}

pub(crate) fn backfill_opencode_server_api_if_configured(
    home: &Path,
    opencode_root: &Path,
    since: Option<&str>,
) -> nabu_core::Result<BackfillReport> {
    let Some(server_url) = opencode_server_url(home)? else {
        return Ok(empty_backfill_report());
    };
    let session_ids = discover_opencode_session_ids(opencode_root)?;
    if session_ids.is_empty() {
        return Ok(empty_backfill_report());
    }

    let _ = since;
    let mut report = empty_backfill_report();
    let session_ids = session_ids.into_iter().collect::<Vec<_>>();
    for chunk in session_ids.chunks(OPENCODE_RECONCILE_FETCH_CONCURRENCY) {
        let fetches = chunk
            .iter()
            .map(|session_id| {
                let server_url = server_url.clone();
                let session_id = session_id.clone();
                std::thread::spawn(move || {
                    let result =
                        opencode_http::fetch_opencode_session_messages(&server_url, &session_id);
                    (session_id, result)
                })
            })
            .collect::<Vec<_>>();

        for fetch in fetches {
            let (session_id, result) = fetch.join().map_err(|_| {
                Error::Validation("OpenCode server reconciliation worker panicked".to_string())
            })?;
            match result {
                Ok(payload) => {
                    let ingest_report =
                        ingest_opencode_server_messages(home, &session_id, payload)?;
                    report.source_files += 1;
                    report.appended_events += ingest_report.appended_events;
                }
                Err(error) => {
                    eprintln!(
                        "warning: skipped OpenCode server reconciliation for session {}: {}",
                        session_id, error
                    );
                }
            }
        }
    }

    Ok(report)
}

fn discover_opencode_session_ids(root: &Path) -> nabu_core::Result<BTreeSet<String>> {
    let mut session_ids = BTreeSet::new();
    let session_root = root.join("storage").join("session");
    if session_root.exists() {
        collect_opencode_session_ids_from_json(&session_root, &mut session_ids)?;
    }
    let message_root = root.join("storage").join("message");
    if message_root.exists() {
        for entry in fs::read_dir(&message_root).map_err(|source| Error::Io {
            path: message_root.clone(),
            source,
        })? {
            let entry = entry.map_err(|source| Error::Io {
                path: message_root.clone(),
                source,
            })?;
            if entry.path().is_dir() {
                if let Ok(session_id) = entry.file_name().into_string() {
                    if !session_id.is_empty() {
                        session_ids.insert(session_id);
                    }
                }
            }
        }
    }
    Ok(session_ids)
}

fn collect_opencode_session_ids_from_json(
    dir: &Path,
    session_ids: &mut BTreeSet<String>,
) -> nabu_core::Result<()> {
    for entry in fs::read_dir(dir).map_err(|source| Error::Io {
        path: dir.to_path_buf(),
        source,
    })? {
        let entry = entry.map_err(|source| Error::Io {
            path: dir.to_path_buf(),
            source,
        })?;
        let path = entry.path();
        if path.is_dir() {
            collect_opencode_session_ids_from_json(&path, session_ids)?;
            continue;
        }
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let file = File::open(&path).map_err(|source| Error::Io {
            path: path.clone(),
            source,
        })?;
        let payload: Value = serde_json::from_reader(BufReader::new(file))?;
        if let Some(session_id) = payload
            .get("id")
            .or_else(|| payload.get("sessionID"))
            .or_else(|| payload.get("session_id"))
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
        {
            session_ids.insert(session_id.to_string());
        }
    }
    Ok(())
}
