//! Terminal/JSON progress rendering for the long-running CLI flows.
//!
//! `ProgressEmitter` is the single sink for backfill, index, and embedding
//! progress. In JSON mode it prints one structured line per event; otherwise it
//! writes human status to stderr. The wizard drives its own UI and passes a
//! quiet emitter so the shared helpers stay silent. The render_* methods are
//! split from the emit_* methods so they can be asserted in tests. `Eta` and
//! `EmbeddingPlanMessage` are private Display newtypes used by those renderers.

use crate::render::ByteSize;
use nabu_core::{BackfillProgress, EmbeddingIndexProgress, EmbeddingModelDisclosure};
use serde_json::json;
use std::borrow::Cow;
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

#[derive(Debug, Clone, Copy)]
pub(crate) struct ProgressEmitter {
    json: bool,
    /// Suppress all output. The wizard drives its own progress UI through the
    /// `Prompter`, so it passes a quiet emitter to the shared backfill helpers
    /// instead of letting them write telemetry straight to the terminal.
    quiet: bool,
}

impl ProgressEmitter {
    pub(crate) fn new(json: bool) -> Self {
        Self { json, quiet: false }
    }

    /// An emitter that writes nothing. Used by the wizard.
    pub(crate) fn quiet() -> Self {
        Self {
            json: false,
            quiet: true,
        }
    }

    pub(crate) fn render(
        self,
        operation: &str,
        phase: &str,
        status: &str,
        processed: usize,
        total: Option<usize>,
        message: &str,
    ) -> String {
        if self.json {
            let timestamp = OffsetDateTime::now_utc()
                .format(&Rfc3339)
                .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_string());
            json!({
                "schema_version": 1,
                "operation": operation,
                "phase": phase,
                "status": status,
                "processed": processed,
                "total": total,
                "unit": "items",
                "message": message,
                "timestamp": timestamp
            })
            .to_string()
        } else {
            match total {
                Some(total) => format!(
                    "progress operation={} phase={} status={} items={}/{} message={}",
                    operation, phase, status, processed, total, message
                ),
                None => format!(
                    "progress operation={} phase={} status={} items={} message={}",
                    operation, phase, status, processed, message
                ),
            }
        }
    }

    fn render_message(
        self,
        operation: &str,
        phase: &str,
        status: &str,
        processed: usize,
        total: Option<usize>,
        message: impl std::fmt::Display,
    ) -> String {
        if self.json {
            let message = message.to_string();
            let timestamp = OffsetDateTime::now_utc()
                .format(&Rfc3339)
                .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_string());
            json!({
                "schema_version": 1,
                "operation": operation,
                "phase": phase,
                "status": status,
                "processed": processed,
                "total": total,
                "unit": "items",
                "message": message,
                "timestamp": timestamp
            })
            .to_string()
        } else {
            match total {
                Some(total) => format!(
                    "progress operation={} phase={} status={} items={}/{} message={}",
                    operation, phase, status, processed, total, message
                ),
                None => format!(
                    "progress operation={} phase={} status={} items={} message={}",
                    operation, phase, status, processed, message
                ),
            }
        }
    }

    pub(crate) fn emit(
        self,
        operation: &str,
        phase: &str,
        status: &str,
        processed: usize,
        total: Option<usize>,
        message: &str,
    ) {
        if self.quiet {
            return;
        }
        eprintln!(
            "{}",
            self.render(operation, phase, status, processed, total, message)
        );
    }

    pub(crate) fn emit_backfill(self, progress: BackfillProgress) {
        if self.quiet {
            return;
        }
        if progress.processed_files != 0
            && progress.processed_files != progress.total_files
            && !progress.processed_files.is_multiple_of(50)
        {
            return;
        }
        if self.json {
            let timestamp = OffsetDateTime::now_utc()
                .format(&Rfc3339)
                .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_string());
            eprintln!(
                "{}",
                json!({
                    "schema_version": 1,
                    "operation": progress.operation,
                    "phase": "backfill",
                    "status": if progress.processed_files == progress.total_files { "completed" } else { "running" },
                    "processed": progress.processed_files,
                    "total": progress.total_files,
                    "unit": "files",
                    "message": "backfill progress",
                    "timestamp": timestamp,
                    "tool": progress.tool,
                    "source_path": progress.source_path,
                    "source_root": progress.source_root
                })
            );
        } else {
            if let Some(path) = progress.source_path.as_deref() {
                eprintln!(
                    "progress operation={} tool={} files={}/{} root={} path={}",
                    progress.operation,
                    progress.tool,
                    progress.processed_files,
                    progress.total_files,
                    progress.source_root,
                    path
                );
            } else {
                eprintln!(
                    "progress operation={} tool={} files={}/{} root={}",
                    progress.operation,
                    progress.tool,
                    progress.processed_files,
                    progress.total_files,
                    progress.source_root
                );
            }
        }
    }

    pub(crate) fn render_embedding_index(self, progress: &EmbeddingIndexProgress) -> String {
        let status = progress.status.as_str();
        let is_plan = progress.phase == "embedding_plan";
        if self.json {
            let message: Cow<'static, str> = if is_plan {
                Cow::Owned(format!("{}", EmbeddingPlanMessage(progress.total_units)))
            } else {
                Cow::Borrowed("embedding index progress")
            };
            let timestamp = OffsetDateTime::now_utc()
                .format(&Rfc3339)
                .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_string());
            json!({
                "schema_version": 1,
                "operation": "embed.index",
                "phase": progress.phase,
                "status": status,
                "processed": progress.embedded_units,
                "total": progress.total_units,
                "unit": "units",
                "message": message,
                "timestamp": timestamp,
                "units_per_second": progress.units_per_second,
                "eta_seconds": progress.eta_seconds,
                "batch_size": progress.batch_size,
                "write_chunk_size": progress.write_chunk_size,
                "intra_threads": progress.intra_threads
            })
            .to_string()
        } else {
            let rendered = format!(
                "progress operation=embed.index phase={} status={} units={}/{} rate={:.1}/s eta={} threads={} batch={} write_chunk={}",
                progress.phase,
                status,
                progress.embedded_units,
                progress.total_units,
                progress.units_per_second,
                Eta(progress.eta_seconds),
                progress.intra_threads,
                progress.batch_size,
                progress.write_chunk_size
            );
            if is_plan {
                format!(
                    "{rendered} message={}",
                    EmbeddingPlanMessage(progress.total_units)
                )
            } else {
                rendered
            }
        }
    }

    pub(crate) fn emit_embedding_index(self, progress: EmbeddingIndexProgress) {
        if self.quiet {
            return;
        }
        eprintln!("{}", self.render_embedding_index(&progress));
    }

    pub(crate) fn render_embedding_download_disclosure(
        self,
        disclosure: &EmbeddingModelDisclosure,
    ) -> Vec<String> {
        let mut rendered = Vec::with_capacity(3);
        rendered.push(self.render_message(
            "embed.download",
            "model_disclosure",
            "info",
            usize::from(disclosure.model_present),
            Some(disclosure.total_files),
            format_args!(
                "model {} from {}",
                disclosure.model_id, disclosure.repository
            ),
        ));
        rendered.push(self.render(
            "embed.download",
            "model_disclosure",
            "info",
            usize::from(disclosure.model_present),
            Some(disclosure.total_files),
            &disclosure.license_summary,
        ));
        rendered.push(self.render_message(
            "embed.download",
            "model_disclosure",
            "info",
            usize::from(disclosure.model_present),
            Some(disclosure.total_files),
            format_args!(
                "measured local footprint at {}: {} across {} expected files",
                disclosure.cache_path,
                ByteSize(disclosure.current_on_disk_bytes),
                disclosure.total_files
            ),
        ));
        rendered
    }

    pub(crate) fn emit_embedding_download_disclosure(self, disclosure: &EmbeddingModelDisclosure) {
        if self.quiet {
            return;
        }
        for line in self.render_embedding_download_disclosure(disclosure) {
            eprintln!("{line}");
        }
    }
}

struct EmbeddingPlanMessage(usize);

impl std::fmt::Display for EmbeddingPlanMessage {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "embedding will process {} unembedded units; one-time CPU-intensive local pass; use --no-embed to build FTS only",
            self.0
        )
    }
}

struct Eta(Option<u64>);

impl std::fmt::Display for Eta {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Some(seconds) = self.0 else {
            return formatter.write_str("unknown");
        };
        let minutes = seconds / 60;
        let seconds = seconds % 60;
        if minutes == 0 {
            write!(formatter, "{seconds}s")
        } else {
            write!(formatter, "{minutes}m{seconds:02}s")
        }
    }
}
