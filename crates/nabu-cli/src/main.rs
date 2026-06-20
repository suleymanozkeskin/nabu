mod jsonc_edit;
mod wizard;

#[cfg(test)]
mod testsupport;

use clap::{Parser, Subcommand, ValueEnum};
use nabu_adapters::{
    claude_status, codex_status, install_claude, install_codex, install_opencode, opencode_status,
    uninstall_claude, uninstall_codex, uninstall_opencode, ConfigChangeReport,
};
#[cfg(test)]
use nabu_core::index_once;
use nabu_core::{
    backfill_dry_run_with_progress, backfill_since_with_progress, canonical_raw_path, dedupe_key,
    doctor_with_options, download_embedding_model_with_progress, embedding_model_disclosure,
    embedding_model_status, export_session_jsonl_with_options,
    export_session_markdown_with_options, index_once_with_options_and_progress, ingest_file,
    ingest_hook_event, ingest_opencode_server_messages, init_home, latest_event,
    opencode_server_url, prune_embedding_cache, purge_all, purge_before, purge_session,
    resolve_home, sanitize_session_id, search_history_page, BackfillProgress, BackfillReport,
    CanonicalType, Corroboration, DedupeParts, EmbeddingIndexProgress, EmbeddingModelDisclosure,
    Error, EventEnvelope, IndexOptions, PurgeAction, PurgeAllOptions, PurgeAllReport, PurgeTier,
    SearchMode, SearchOptions, SessionOptions, Source, Tool, SCHEMA_VERSION,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::borrow::Cow;
use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Cursor, IsTerminal, Read, Seek, SeekFrom, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Stdio};
use std::str::FromStr;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

const OPENCODE_RECONCILE_FETCH_CONCURRENCY: usize = 8;

#[derive(Debug, Parser)]
#[command(name = "nabu", version, about = "Local coding-agent history keeper")]
struct Cli {
    #[arg(long, global = true, value_name = "PATH")]
    home: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

impl Cli {
    fn renders_errors_as_json(&self) -> bool {
        self.command.renders_errors_as_json()
    }
}

#[derive(Debug, Subcommand)]
enum Command {
    Init,
    /// Guided first-run and management front end over the explicit commands.
    Wizard,
    Ingest {
        #[command(subcommand)]
        command: IngestCommand,
    },
    Index {
        #[arg(long)]
        once: bool,
        #[arg(long)]
        watch: bool,
        #[arg(long)]
        no_embed: bool,
        #[arg(long)]
        json_progress: bool,
    },
    Backfill {
        #[arg(long)]
        tool: BackfillTool,
        #[arg(long)]
        path: Option<PathBuf>,
        #[arg(long)]
        since: Option<String>,
        #[arg(long)]
        dry_run: bool,
        #[arg(long)]
        json_progress: bool,
    },
    Embed {
        #[command(subcommand)]
        command: EmbedCommand,
    },
    Bench {
        #[command(subcommand)]
        command: BenchCommand,
    },
    Install {
        tool: AgentTool,
        #[arg(long)]
        dry_run: bool,
    },
    Uninstall {
        tool: AgentTool,
        #[arg(long)]
        dry_run: bool,
    },
    Search {
        query: String,
        #[arg(long)]
        tool: Option<Tool>,
        #[arg(long)]
        session: Option<String>,
        #[arg(long)]
        cwd: Option<String>,
        #[arg(long)]
        since: Option<String>,
        #[arg(long = "type")]
        canonical_type: Option<String>,
        #[arg(long)]
        file: Option<String>,
        #[arg(long)]
        command: Option<String>,
        #[arg(long, value_enum, default_value_t = SearchModeArg::Auto)]
        mode: SearchModeArg,
        #[arg(long)]
        corroborate: bool,
        #[arg(long, default_value_t = 10)]
        limit: usize,
        #[arg(long, default_value_t = 0)]
        offset: usize,
        #[arg(long)]
        full: bool,
        #[arg(long)]
        include_deltas: bool,
        #[arg(long)]
        no_dedupe: bool,
        #[arg(long, default_value_t = 240)]
        max_snippet_chars: usize,
        #[arg(long, value_enum, default_value_t = OutputFormat::Human)]
        format: OutputFormat,
    },
    Show {
        tool: Tool,
        session_id: String,
        #[arg(long, default_value_t = 100)]
        limit_events: usize,
        #[arg(long)]
        after_raw_line: Option<i64>,
        #[arg(long)]
        around_line: Option<i64>,
        #[arg(long, default_value_t = 5)]
        before: usize,
        #[arg(long, default_value_t = 5)]
        after: usize,
        #[arg(long = "type")]
        canonical_type: Option<String>,
        #[arg(long)]
        include_deltas: bool,
        #[arg(long)]
        corroborate: bool,
        #[arg(long, value_enum, default_value_t = OutputFormat::Human)]
        format: OutputFormat,
    },
    Tail {
        tool: Tool,
        session_id: String,
        #[arg(long)]
        follow: bool,
    },
    Export {
        tool: Tool,
        session_id: String,
        #[arg(long, value_enum)]
        format: ExportFormat,
        #[arg(long)]
        redact: bool,
    },
    Purge {
        #[arg(long)]
        session: Option<String>,
        #[arg(long)]
        before: Option<String>,
        /// Remove every nabu artifact: uninstall hooks from all tools and
        /// delete the store. Previews and asks for confirmation first.
        #[arg(long)]
        all: bool,
        /// With --all: keep the downloaded embedding model under models/.
        #[arg(long, requires = "all")]
        keep_model: bool,
        /// With --all: keep config.toml (your settings).
        #[arg(long, requires = "all")]
        keep_config: bool,
        /// With --all: show exactly what would be removed, then exit without deleting.
        #[arg(long, requires = "all")]
        dry_run: bool,
        /// With --all: skip the typed confirmation (required in non-interactive use).
        #[arg(long, requires = "all")]
        yes: bool,
    },
    Doctor {
        #[arg(long, value_enum, default_value_t = DoctorTool::All)]
        tool: DoctorTool,
        #[arg(long)]
        deep: bool,
        #[arg(long)]
        json: bool,
    },
    Mcp {
        #[command(subcommand)]
        command: McpCommand,
    },
}

impl Command {
    fn renders_errors_as_json(&self) -> bool {
        match self {
            Command::Search { format, .. } | Command::Show { format, .. } => {
                *format == OutputFormat::Json
            }
            Command::Doctor { json, .. } => *json,
            Command::Mcp {
                command: McpCommand::Validate { json, .. },
            } => *json,
            _ => false,
        }
    }
}

#[derive(Debug, Subcommand)]
enum IngestCommand {
    Hook {
        #[arg(long)]
        tool: Tool,
    },
    File {
        #[arg(long)]
        tool: Tool,
        #[arg(long, value_enum)]
        source: IngestSource,
        #[arg(long)]
        path: PathBuf,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum IngestSource {
    #[value(name = "backfill")]
    Backfill,
    #[value(name = "exec_json")]
    ExecJson,
    #[value(name = "app_server")]
    AppServer,
    #[value(name = "event_stream")]
    EventStream,
    #[value(name = "transcript_tail")]
    TranscriptTail,
}

impl IngestSource {
    fn source(self) -> Source {
        match self {
            IngestSource::Backfill => Source::Backfill,
            IngestSource::ExecJson => Source::ExecJson,
            IngestSource::AppServer => Source::AppServer,
            IngestSource::EventStream => Source::EventStream,
            IngestSource::TranscriptTail => Source::TranscriptTail,
        }
    }
}

#[derive(Debug, Subcommand)]
enum BenchCommand {
    Ingest {
        #[arg(long)]
        events: PathBuf,
        #[arg(long, default_value_t = 0)]
        seed_events: usize,
        #[arg(long, default_value_t = 1000)]
        iterations: usize,
        #[arg(long)]
        json_progress: bool,
    },
    Search {
        #[arg(long)]
        query: String,
        #[arg(long, default_value_t = 100)]
        iterations: usize,
        #[arg(long, default_value_t = 10)]
        limit: usize,
        #[arg(long)]
        json_progress: bool,
    },
}

#[derive(Debug, Subcommand)]
enum EmbedCommand {
    Status,
    Download {
        #[arg(long, default_value = "embeddinggemma-300m-q4")]
        model: String,
        #[arg(long)]
        yes: bool,
        #[arg(long)]
        json_progress: bool,
    },
    Prune {
        #[arg(long)]
        yes: bool,
        #[arg(long)]
        json_progress: bool,
    },
}

#[derive(Debug, Subcommand)]
enum McpCommand {
    Serve {
        #[arg(long, value_enum)]
        transport: McpTransport,
    },
    Install {
        tool: AgentTool,
        #[arg(long)]
        dry_run: bool,
    },
    Uninstall {
        tool: AgentTool,
        #[arg(long)]
        dry_run: bool,
    },
    Validate {
        tool: AgentTool,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum McpTransport {
    Stdio,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum BackfillTool {
    Codex,
    Claude,
    Opencode,
    All,
}

impl BackfillTool {
    fn selection(self) -> Option<Tool> {
        match self {
            BackfillTool::Codex => Some(Tool::Codex),
            BackfillTool::Claude => Some(Tool::Claude),
            BackfillTool::Opencode => Some(Tool::Opencode),
            BackfillTool::All => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum AgentTool {
    Codex,
    Claude,
    Opencode,
    All,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum DoctorTool {
    Codex,
    Claude,
    Opencode,
    All,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum OutputFormat {
    Human,
    Json,
    Markdown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum SearchModeArg {
    Auto,
    Lexical,
    Hybrid,
}

impl From<SearchModeArg> for SearchMode {
    fn from(value: SearchModeArg) -> Self {
        match value {
            SearchModeArg::Auto => SearchMode::Auto,
            SearchModeArg::Lexical => SearchMode::Lexical,
            SearchModeArg::Hybrid => SearchMode::Hybrid,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ExportFormat {
    Jsonl,
    Markdown,
}

fn main() {
    let cli = Cli::parse();
    let render_errors_as_json = cli.renders_errors_as_json();
    if let Err(error) = run(cli) {
        let exit_code = cli_exit_code(&error);
        if render_errors_as_json {
            println!(
                "{}",
                serde_json::to_string_pretty(&json_error(&error)).unwrap_or_else(|_| {
                    "{\"ok\":false,\"error\":{\"code\":\"INTERNAL_ERROR\",\"message\":\"failed to render error\",\"recoverable\":false,\"hint\":\"Retry with human output.\",\"details\":{}}}".to_string()
                })
            );
        } else {
            eprintln!("{error}");
        }
        std::process::exit(exit_code);
    }
}

fn run(cli: Cli) -> nabu_core::Result<()> {
    let Cli { home, command } = cli;
    let home = resolve_home(home)?;

    match command {
        Command::Init => {
            let report = init_home(&home)?;
            println!("initialized {}", report.home.display());
        }
        Command::Wizard => {
            wizard::run_wizard(&home)?;
        }
        Command::Ingest {
            command: IngestCommand::Hook { tool },
        } => {
            let mut input = String::new();
            std::io::stdin()
                .read_to_string(&mut input)
                .map_err(|source| Error::Io {
                    path: PathBuf::from("<stdin>"),
                    source,
                })?;
            let payload: Value = serde_json::from_str(&input)?;
            match ingest_hook_event(&home, tool, payload) {
                Ok(report) => {
                    if report.appended {
                        println!(
                            "appended {} at offset {}",
                            report.raw_file.display(),
                            report.raw_offset
                        );
                    } else {
                        println!(
                            "skipped duplicate {} at offset {}",
                            report.raw_file.display(),
                            report.raw_offset
                        );
                    }
                }
                Err(error @ Error::Io { .. }) => {
                    eprintln!("{error}");
                }
                Err(error) => return Err(error),
            }
        }
        Command::Ingest {
            command: IngestCommand::File { tool, source, path },
        } => {
            let report = ingest_file(&home, tool, source.source(), &path)?;
            println!(
                "ingested {} events from {}",
                report.appended_events,
                path.display()
            );
        }
        Command::Index {
            once,
            watch,
            no_embed,
            json_progress,
        } => {
            let progress = ProgressEmitter::new(json_progress);
            let index_options = IndexOptions { embed: !no_embed };
            if watch {
                loop {
                    progress.emit("index", "index", "started", 0, None, "starting index pass");
                    let report =
                        index_once_with_options_and_progress(&home, index_options, |event| {
                            progress.emit_embedding_index(event)
                        })?;
                    progress.emit(
                        "index",
                        "index",
                        "completed",
                        report.indexed_events,
                        None,
                        "index pass completed",
                    );
                    std::thread::sleep(Duration::from_secs(2));
                }
            }
            if !once {
                return Err(Error::Validation(
                    "index requires --once or --watch".to_string(),
                ));
            }
            progress.emit("index", "index", "started", 0, None, "starting index pass");
            let report = index_once_with_options_and_progress(&home, index_options, |event| {
                progress.emit_embedding_index(event)
            })?;
            progress.emit(
                "index",
                "index",
                "completed",
                report.indexed_events,
                None,
                "index pass completed",
            );
            println!("indexed {} new events", report.indexed_events);
        }
        Command::Backfill {
            tool,
            path,
            since,
            dry_run,
            json_progress,
        } => {
            if dry_run {
                let report = run_backfill_dry_run_command(
                    &home,
                    tool,
                    path.as_deref(),
                    since.as_deref(),
                    ProgressEmitter::new(json_progress),
                )?;
                println!("{}", serde_json::to_string_pretty(&report)?);
                return Ok(());
            }
            let report = run_backfill_command(
                &home,
                tool,
                path.as_deref(),
                since.as_deref(),
                ProgressEmitter::new(json_progress),
            )?;
            println!(
                "backfilled {} source files, appended {} events, wrote {} checkpoints, emitted {} discontinuities",
                report.source_files,
                report.appended_events,
                report.checkpoint_files,
                report.discontinuities
            );
        }
        Command::Embed { command } => match command {
            EmbedCommand::Status => {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json_success(serde_json::to_value(
                        embedding_model_status(&home)
                    )?))?
                );
            }
            EmbedCommand::Download {
                model,
                yes,
                json_progress,
            } => {
                let progress = ProgressEmitter::new(json_progress);
                progress.emit(
                    "embed.download",
                    "model_acquisition",
                    "started",
                    0,
                    Some(1),
                    "checking model acquisition request",
                );
                if model != "embeddinggemma-300m-q4" {
                    progress.emit(
                        "embed.download",
                        "model_acquisition",
                        "failed",
                        0,
                        Some(1),
                        "unsupported model",
                    );
                    return Err(Error::Validation(format!(
                        "unsupported embedding model: {model}"
                    )));
                }
                let disclosure = embedding_model_disclosure(&home, &model)?;
                progress.emit_embedding_download_disclosure(&disclosure);
                if !yes {
                    progress.emit(
                        "embed.download",
                        "model_acquisition",
                        "failed",
                        0,
                        Some(1),
                        "explicit consent required",
                    );
                    return Err(Error::Validation(
                        "embed download requires --yes after reviewing the printed model license and measured footprint"
                            .to_string(),
                    ));
                }
                let report =
                    download_embedding_model_with_progress(&home, &model, |download_progress| {
                        progress.emit(
                            "embed.download",
                            "model_acquisition",
                            &download_progress.phase,
                            download_progress.downloaded_files,
                            Some(download_progress.total_files),
                            &download_progress.file,
                        );
                    })?;
                progress.emit(
                    "embed.download",
                    "model_acquisition",
                    "completed",
                    report.downloaded_files,
                    Some(report.total_files),
                    "model cache ready",
                );
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json_success(serde_json::to_value(report)?))?
                );
            }
            EmbedCommand::Prune { yes, json_progress } => {
                let progress = ProgressEmitter::new(json_progress);
                progress.emit(
                    "embed.prune",
                    "model_cache",
                    "started",
                    0,
                    Some(1),
                    "preparing model-cache prune",
                );
                if !yes {
                    progress.emit(
                        "embed.prune",
                        "model_cache",
                        "failed",
                        0,
                        Some(1),
                        "explicit consent required",
                    );
                    return Err(Error::Validation(
                        "embed prune requires --yes because it removes local model cache files"
                            .to_string(),
                    ));
                }
                let footprint = prune_embedding_cache(&home)?;
                progress.emit(
                    "embed.prune",
                    "model_cache",
                    "completed",
                    1,
                    Some(1),
                    "model cache pruned",
                );
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json_success(serde_json::to_value(footprint)?))?
                );
            }
        },
        Command::Bench { command } => match command {
            BenchCommand::Ingest {
                events,
                seed_events,
                iterations,
                json_progress,
            } => {
                let progress = ProgressEmitter::new(json_progress);
                progress.emit(
                    "bench.ingest",
                    "bench",
                    "started",
                    0,
                    Some(iterations),
                    "starting ingest benchmark",
                );
                let report = run_ingest_bench(&events, seed_events, iterations)?;
                progress.emit(
                    "bench.ingest",
                    "bench",
                    "completed",
                    iterations,
                    Some(iterations),
                    "ingest benchmark completed",
                );
                println!("{}", serde_json::to_string_pretty(&report)?);
            }
            BenchCommand::Search {
                query,
                iterations,
                limit,
                json_progress,
            } => {
                let progress = ProgressEmitter::new(json_progress);
                progress.emit(
                    "bench.search",
                    "bench",
                    "started",
                    0,
                    Some(iterations),
                    "starting search benchmark",
                );
                let report = run_search_bench(&home, &query, iterations, limit)?;
                progress.emit(
                    "bench.search",
                    "bench",
                    "completed",
                    iterations,
                    Some(iterations),
                    "search benchmark completed",
                );
                println!("{}", serde_json::to_string_pretty(&report)?);
            }
        },
        Command::Install { tool, dry_run } => match tool {
            AgentTool::Claude => {
                let report = install_claude(&home, dry_run)?;
                println!("{}", report.summary);
                println!("target: {}", report.target_path.display());
                println!("{}", report.diff);
            }
            AgentTool::All => {
                for report in [
                    install_codex(&home, dry_run)?,
                    install_claude(&home, dry_run)?,
                    install_opencode(&home, dry_run)?,
                ] {
                    println!("{}", report.summary);
                    println!("target: {}", report.target_path.display());
                    println!("{}", report.diff);
                }
            }
            AgentTool::Opencode => {
                let report = install_opencode(&home, dry_run)?;
                println!("{}", report.summary);
                println!("target: {}", report.target_path.display());
                println!("{}", report.diff);
            }
            AgentTool::Codex => {
                let report = install_codex(&home, dry_run)?;
                println!("{}", report.summary);
                println!("target: {}", report.target_path.display());
                println!("{}", report.diff);
            }
        },
        Command::Uninstall { tool, dry_run } => match tool {
            AgentTool::Claude => {
                let report = uninstall_claude(&home, dry_run)?;
                println!("{}", report.summary);
                println!("target: {}", report.target_path.display());
                println!("{}", report.diff);
            }
            AgentTool::All => {
                for report in [
                    uninstall_codex(&home, dry_run)?,
                    uninstall_claude(&home, dry_run)?,
                    uninstall_opencode(&home, dry_run)?,
                ] {
                    println!("{}", report.summary);
                    println!("target: {}", report.target_path.display());
                    println!("{}", report.diff);
                }
            }
            AgentTool::Opencode => {
                let report = uninstall_opencode(&home, dry_run)?;
                println!("{}", report.summary);
                println!("target: {}", report.target_path.display());
                println!("{}", report.diff);
            }
            AgentTool::Codex => {
                let report = uninstall_codex(&home, dry_run)?;
                println!("{}", report.summary);
                println!("target: {}", report.target_path.display());
                println!("{}", report.diff);
            }
        },
        Command::Search {
            query,
            tool,
            session,
            cwd,
            since,
            canonical_type,
            file,
            command,
            mode,
            corroborate,
            limit,
            offset,
            full,
            include_deltas,
            no_dedupe,
            max_snippet_chars,
            format,
        } => {
            let page = search_history_page(
                &home,
                &query,
                SearchOptions {
                    tool,
                    session_id: session,
                    cwd,
                    since,
                    canonical_type,
                    file,
                    command,
                    limit,
                    offset,
                    include_payload: full,
                    include_deltas,
                    dedupe: !no_dedupe,
                    max_snippet_chars,
                    mode: mode.into(),
                    corroborate,
                },
            )?;
            match format {
                OutputFormat::Json => {
                    println!("{}", serde_json::to_string_pretty(&page)?);
                }
                OutputFormat::Markdown => {
                    println!(
                        "returned: {}  truncated: {}  next_offset: {}  max_snippet_chars_applied: {}",
                        page.returned,
                        page.truncated,
                        OptionalValue(page.continuation.as_ref().map(|continuation| continuation.next_offset)),
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
        }
        Command::Show {
            tool,
            session_id,
            limit_events,
            after_raw_line,
            around_line,
            before,
            after,
            canonical_type,
            include_deltas,
            corroborate,
            format,
        } => {
            let page = nabu_core::get_session_page(
                &home,
                tool,
                &session_id,
                SessionOptions {
                    limit_events,
                    after_raw_line,
                    around_raw_line: around_line,
                    before,
                    after,
                    include_deltas,
                    canonical_type,
                    redact: false,
                    corroborate,
                },
            )?;
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
        }
        Command::Tail {
            tool,
            session_id,
            follow,
        } => {
            if follow {
                follow_session_jsonl(&home, tool, &session_id)?;
            } else {
                print!(
                    "{}",
                    export_session_jsonl_with_options(&home, tool, &session_id, false)?
                );
            }
        }
        Command::Export {
            tool,
            session_id,
            format,
            redact,
        } => match format {
            ExportFormat::Jsonl => {
                print!(
                    "{}",
                    export_session_jsonl_with_options(&home, tool, &session_id, redact)?
                );
            }
            ExportFormat::Markdown => {
                print!(
                    "{}",
                    export_session_markdown_with_options(&home, tool, &session_id, redact)?
                );
            }
        },
        Command::Purge {
            session,
            before,
            all,
            keep_model,
            keep_config,
            dry_run,
            yes,
        } => {
            if all {
                if session.is_some() || before.is_some() {
                    return Err(Error::Validation(
                        "`--all` cannot be combined with --session or --before".to_string(),
                    ));
                }
                run_purge_all(&home, keep_model, keep_config, dry_run, yes)?;
            } else {
                let report = match (session, before) {
                    (Some(session), None) => {
                        let (tool, session_id) = parse_session_selector(&session)?;
                        purge_session(&home, tool, session_id)?
                    }
                    (None, Some(before)) => purge_before(&home, &before)?,
                    _ => {
                        return Err(Error::Validation(
                            "purge requires exactly one of --session TOOL:SESSION_ID, --before DATE_OR_DURATION, or --all"
                                .to_string(),
                        ))
                    }
                };
                println!("{}", serde_json::to_string_pretty(&report)?);
            }
        }
        Command::Doctor {
            tool,
            deep,
            json: as_json,
        } => {
            if as_json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json_success(doctor_json_data(
                        &home, tool, deep
                    )?))?
                );
            } else {
                let report = doctor_with_options(&home, deep);
                println!("level={}", report.level);
                println!("integrity={}", report.integrity);
                println!("storage.ok={}", report.storage.ok);
                println!("index.ok={}", report.index.ok);
                println!("backfill.ok={}", report.backfill.ok);
                print_tool_doctor_human(&home, tool)?;
            }
        }
        Command::Mcp { command } => match command {
            McpCommand::Serve { transport } => match transport {
                McpTransport::Stdio => nabu_mcp::serve_stdio(home)?,
            },
            McpCommand::Install { tool, dry_run } => {
                for report in mcp_apply_all(&home, tool, McpConfigAction::Install, dry_run)? {
                    println!("{}", report.summary);
                    println!("target: {}", report.target_path.display());
                    println!("{}", report.diff);
                }
            }
            McpCommand::Uninstall { tool, dry_run } => {
                for report in mcp_apply_all(&home, tool, McpConfigAction::Uninstall, dry_run)? {
                    println!("{}", report.summary);
                    println!("target: {}", report.target_path.display());
                    println!("{}", report.diff);
                }
            }
            McpCommand::Validate {
                tool,
                json: as_json,
            } => {
                let value = mcp_validate_all(&home, tool)?;
                if as_json {
                    println!("{}", serde_json::to_string_pretty(&json_success(value))?);
                } else {
                    println!("{}", value);
                }
            }
        },
    }

    Ok(())
}

fn doctor_json_data(home: &Path, tool: DoctorTool, deep: bool) -> nabu_core::Result<Value> {
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

fn json_success(data: Value) -> Value {
    json!({
        "ok": true,
        "data": data
    })
}

fn json_error(error: &Error) -> Value {
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

fn cli_exit_code(error: &Error) -> i32 {
    match cli_error_code(error) {
        "VALIDATION_ERROR" | "NOT_FOUND" => 1,
        "STORAGE_UNAVAILABLE" | "INDEX_UNAVAILABLE" | "PERMISSION_DENIED" => 3,
        _ => 5,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum McpConfigAction {
    Install,
    Uninstall,
}

fn parse_session_selector(selector: &str) -> nabu_core::Result<(Tool, &str)> {
    let Some((tool, session_id)) = selector.split_once(':') else {
        return Err(Error::Validation(
            "--session must use TOOL:SESSION_ID".to_string(),
        ));
    };
    if session_id.is_empty() {
        return Err(Error::Validation(
            "session id must not be empty".to_string(),
        ));
    }
    Ok((Tool::from_str(tool)?, session_id))
}

#[derive(Debug, Clone, Copy)]
struct ProgressEmitter {
    json: bool,
    /// Suppress all output. The wizard drives its own progress UI through the
    /// `Prompter`, so it passes a quiet emitter to the shared backfill helpers
    /// instead of letting them write telemetry straight to the terminal.
    quiet: bool,
}

impl ProgressEmitter {
    fn new(json: bool) -> Self {
        Self { json, quiet: false }
    }

    /// An emitter that writes nothing. Used by the wizard.
    fn quiet() -> Self {
        Self {
            json: false,
            quiet: true,
        }
    }

    fn render(
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

    fn emit(
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

    fn emit_backfill(self, progress: BackfillProgress) {
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

    fn render_embedding_index(self, progress: &EmbeddingIndexProgress) -> String {
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

    fn emit_embedding_index(self, progress: EmbeddingIndexProgress) {
        if self.quiet {
            return;
        }
        eprintln!("{}", self.render_embedding_index(&progress));
    }

    fn render_embedding_download_disclosure(
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

    fn emit_embedding_download_disclosure(self, disclosure: &EmbeddingModelDisclosure) {
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

fn uninstall_all(home: &Path, dry_run: bool) -> nabu_core::Result<[ConfigChangeReport; 3]> {
    Ok([
        uninstall_codex(home, dry_run)?,
        uninstall_claude(home, dry_run)?,
        uninstall_opencode(home, dry_run)?,
    ])
}

/// Orchestrate a full removal: preview (store + hooks), consent, then execute.
/// The store wipe is the closed-allowlist [`purge_all`]; hook removal reuses the
/// contracted uninstall path. Hooks are removed before the store so capture
/// stops before its data goes.
fn run_purge_all(
    home: &Path,
    keep_model: bool,
    keep_config: bool,
    dry_run: bool,
    yes: bool,
) -> nabu_core::Result<()> {
    let preview = purge_all(
        home,
        PurgeAllOptions {
            keep_model,
            keep_config,
            dry_run: true,
        },
    )?;
    let hook_previews = uninstall_all(home, true)?;
    print_purge_all_preview(&preview, &hook_previews);

    if dry_run {
        println!("\ndry run — nothing was removed.");
        return Ok(());
    }

    if !yes {
        if !std::io::stdin().is_terminal() {
            return Err(Error::Validation(
                "purge --all needs confirmation; re-run with --yes in a non-interactive context"
                    .to_string(),
            ));
        }
        print!("\nType \"purge\" to remove everything above (anything else aborts): ");
        std::io::stdout().flush().map_err(|source| Error::Io {
            path: PathBuf::from("<stdout>"),
            source,
        })?;
        let mut answer = String::new();
        std::io::stdin()
            .read_line(&mut answer)
            .map_err(|source| Error::Io {
                path: PathBuf::from("<stdin>"),
                source,
            })?;
        if answer.trim() != "purge" {
            println!("aborted — nothing removed.");
            return Ok(());
        }
    }

    println!("\nremoving integrations...");
    for report in uninstall_all(home, false)? {
        println!(
            "  {:<9} {}",
            report.tool,
            if report.changed {
                "removed"
            } else {
                "not installed"
            }
        );
    }
    let report = purge_all(
        home,
        PurgeAllOptions {
            keep_model,
            keep_config,
            dry_run: false,
        },
    )?;
    print_purge_all_result(&report);
    Ok(())
}

struct ByteSize(u64);

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

fn print_purge_all_preview(preview: &PurgeAllReport, hooks: &[ConfigChangeReport]) {
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

fn print_purge_all_result(report: &PurgeAllReport) {
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

fn run_backfill_command(
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

fn run_backfill_dry_run_command(
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
        match tool {
            Tool::Codex => {
                let codex_home = codex_home_dir()?;
                let sessions = codex_home.join("sessions");
                visit(Tool::Codex, &sessions)?;
                let archived_sessions = codex_home.join("archived_sessions");
                visit(Tool::Codex, &archived_sessions)?;
            }
            Tool::Claude => {
                let projects = claude_projects_dir()?;
                visit(Tool::Claude, &projects)?;
            }
            Tool::Opencode => {
                let project_data = opencode_project_data_dir()?;
                visit(Tool::Opencode, &project_data)?;
            }
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

fn codex_home_dir() -> nabu_core::Result<PathBuf> {
    if let Some(codex_home) = std::env::var_os("CODEX_HOME") {
        return Ok(PathBuf::from(codex_home));
    }
    let Some(home) = std::env::var_os("HOME") else {
        return Err(Error::HomeUnavailable);
    };
    Ok(PathBuf::from(home).join(".codex"))
}

fn claude_projects_dir() -> nabu_core::Result<PathBuf> {
    if let Some(config_dir) = std::env::var_os("CLAUDE_CONFIG_DIR") {
        return Ok(PathBuf::from(config_dir).join("projects"));
    }
    let Some(home) = std::env::var_os("HOME") else {
        return Err(Error::HomeUnavailable);
    };
    Ok(PathBuf::from(home).join(".claude").join("projects"))
}

fn opencode_project_data_dir() -> nabu_core::Result<PathBuf> {
    let Some(home) = std::env::var_os("HOME") else {
        return Err(Error::HomeUnavailable);
    };
    Ok(PathBuf::from(home)
        .join(".local")
        .join("share")
        .join("opencode"))
}

fn backfill_opencode_server_api_if_configured(
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
                    let result = fetch_opencode_session_messages(&server_url, &session_id);
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

fn fetch_opencode_session_messages(server_url: &str, session_id: &str) -> nabu_core::Result<Value> {
    let (host, port, base_path) = parse_http_url(server_url)?;
    let mut stream = TcpStream::connect((host, port)).map_err(|source| Error::Io {
        path: PathBuf::from(server_url),
        source,
    })?;
    let mut request = String::with_capacity(
        "GET  HTTP/1.1\r\nHost: \r\nAccept: application/json\r\nConnection: close\r\n\r\n".len()
            + opencode_session_messages_path_len(base_path, session_id)
            + host.len(),
    );
    request.push_str("GET ");
    push_opencode_session_messages_path(&mut request, base_path, session_id);
    request.push_str(" HTTP/1.1\r\nHost: ");
    request.push_str(host);
    request.push_str("\r\nAccept: application/json\r\nConnection: close\r\n\r\n");
    stream
        .write_all(request.as_bytes())
        .map_err(|source| Error::Io {
            path: PathBuf::from(server_url),
            source,
        })?;
    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .map_err(|source| Error::Io {
            path: PathBuf::from(server_url),
            source,
        })?;
    parse_http_json_response(server_url, &response)
}

fn parse_http_url(url: &str) -> nabu_core::Result<(&str, u16, &str)> {
    let Some(rest) = url.strip_prefix("http://") else {
        return Err(Error::Validation(
            "OpenCode server URL must use http:// for local reconciliation".to_string(),
        ));
    };
    let (authority, path) = rest.split_once('/').unwrap_or((rest, ""));
    let (host, port) = authority
        .rsplit_once(':')
        .and_then(|(host, port)| Some((host, port.parse::<u16>().ok()?)))
        .unwrap_or((authority, 80));
    if host.is_empty() {
        return Err(Error::Validation(
            "OpenCode server URL host must not be empty".to_string(),
        ));
    }
    Ok((host, port, path))
}

fn opencode_session_messages_path_len(base: &str, session_id: &str) -> usize {
    let base = base.trim_matches('/');
    let suffix_len = "/session/".len() + session_id.len() + "/message".len();
    if base.is_empty() {
        suffix_len
    } else {
        1 + base.len() + suffix_len
    }
}

fn push_opencode_session_messages_path(request: &mut String, base: &str, session_id: &str) {
    let base = base.trim_matches('/');
    if !base.is_empty() {
        request.push('/');
        request.push_str(base);
    }
    request.push_str("/session/");
    request.push_str(session_id);
    request.push_str("/message");
}

fn parse_http_json_response(server_url: &str, response: &[u8]) -> nabu_core::Result<Value> {
    let Some(split) = response.windows(4).position(|window| window == b"\r\n\r\n") else {
        return Err(Error::Validation(
            "OpenCode server returned an invalid HTTP response".to_string(),
        ));
    };
    let headers = std::str::from_utf8(&response[..split]).map_err(|_| {
        Error::Validation("OpenCode server returned non-UTF8 HTTP headers".to_string())
    })?;
    let status = headers.lines().next().unwrap_or_default();
    if !status.contains(" 200 ") {
        return Err(Error::Validation(format!(
            "OpenCode server request failed: {status}"
        )));
    }
    let body = &response[split + 4..];
    serde_json::from_slice(body).map_err(|source| {
        Error::Validation(format!(
            "OpenCode server returned invalid JSON from {server_url}: {source}"
        ))
    })
}

fn mcp_apply_all(
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

fn mcp_apply_one(
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
    let target_path = codex_mcp_config_path()?;
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
    let target_path = claude_mcp_config_path()?;
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

fn mcp_apply_opencode(
    home: &Path,
    action: McpConfigAction,
    dry_run: bool,
) -> nabu_core::Result<ConfigChangeReport> {
    let target_path = opencode_mcp_config_path()?;
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

fn mcp_validate_all(home: &Path, tool: AgentTool) -> nabu_core::Result<Value> {
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

fn codex_mcp_entry_installed() -> bool {
    codex_mcp_config_path()
        .ok()
        .and_then(|path| read_text_or_empty(&path).ok())
        .map(|content| content.contains("[mcp_servers.nabu]"))
        .unwrap_or(false)
}

fn claude_mcp_entry_installed() -> bool {
    claude_mcp_config_path()
        .ok()
        .and_then(|path| read_text_or_empty(&path).ok())
        .and_then(|content| serde_json::from_str::<Value>(&content).ok())
        .and_then(|config| config.pointer("/mcpServers/nabu").cloned())
        .is_some()
}

fn opencode_mcp_entry_installed() -> bool {
    opencode_mcp_config_path()
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

fn codex_mcp_config_path() -> nabu_core::Result<PathBuf> {
    if let Some(codex_home) = std::env::var_os("CODEX_HOME") {
        return Ok(PathBuf::from(codex_home).join("config.toml"));
    }
    let Some(home) = std::env::var_os("HOME") else {
        return Err(Error::HomeUnavailable);
    };
    Ok(PathBuf::from(home).join(".codex").join("config.toml"))
}

fn claude_mcp_config_path() -> nabu_core::Result<PathBuf> {
    let Some(home) = std::env::var_os("HOME") else {
        return Err(Error::HomeUnavailable);
    };
    Ok(PathBuf::from(home).join(".claude.json"))
}

fn opencode_mcp_config_path() -> nabu_core::Result<PathBuf> {
    if let Some(config_dir) = std::env::var_os("OPENCODE_CONFIG_DIR") {
        return Ok(PathBuf::from(config_dir).join("opencode.json"));
    }
    if let Some(config_home) = std::env::var_os("XDG_CONFIG_HOME") {
        return Ok(PathBuf::from(config_home)
            .join("opencode")
            .join("opencode.json"));
    }
    let Some(home) = std::env::var_os("HOME") else {
        return Err(Error::HomeUnavailable);
    };
    Ok(PathBuf::from(home)
        .join(".config")
        .join("opencode")
        .join("opencode.json"))
}

fn add_codex_mcp_block(content: &str) -> String {
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

fn add_claude_mcp(mut config: Value) -> Value {
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

fn add_opencode_mcp(mut config: Value) -> Value {
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

fn read_text_or_empty(path: &PathBuf) -> nabu_core::Result<String> {
    if !path.exists() {
        return Ok(String::new());
    }
    fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.clone(),
        source,
    })
}

fn write_text_config(path: &PathBuf, content: &str, mode: u32) -> nabu_core::Result<()> {
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
    fs::write(path, content).map_err(|source| Error::Io {
        path: path.clone(),
        source,
    })?;
    chmod_path(path, final_mode)
}

fn backup_cli_config(
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

fn text_diff(before: &str, after: &str) -> String {
    let mut diff = String::with_capacity(before.len() + after.len() + 24);
    diff.push_str("--- before\n");
    diff.push_str(before);
    diff.push_str("\n--- after\n");
    diff.push_str(after);
    diff.push('\n');
    diff
}

fn mcp_operation_name(action: McpConfigAction) -> &'static str {
    match action {
        McpConfigAction::Install => "mcp-install",
        McpConfigAction::Uninstall => "mcp-uninstall",
    }
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

fn command_in_path(command: &str) -> bool {
    let Some(paths) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&paths).any(|mut path| {
        path.push(command);
        path.is_file()
    })
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

fn print_tool_doctor_human(home: &Path, tool: DoctorTool) -> nabu_core::Result<()> {
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

struct AlsoAt<'a>(&'a [i64]);

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

struct OptionalValue<T>(Option<T>);

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

fn print_corroboration_human(corroboration: Option<&Corroboration>) {
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

fn print_corroboration_markdown(corroboration: Option<&Corroboration>) {
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

fn run_ingest_bench(
    events_path: &PathBuf,
    seed_events: usize,
    iterations: usize,
) -> nabu_core::Result<Value> {
    let events = load_bench_events(events_path)?;
    if events.is_empty() {
        return Err(Error::Validation(
            "benchmark events file must contain at least one JSONL event".to_string(),
        ));
    }
    if seed_events > 0 && seed_events < 10_000 {
        return Err(Error::Validation(
            "--seed-events must be 0 or at least 10000".to_string(),
        ));
    }

    let home = std::env::temp_dir().join(format!(
        "nabu-bench-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0)
    ));
    init_home(&home)?;

    let iterations = iterations.clamp(1, 10_000);
    if seed_events > 0 {
        return run_large_session_ingest_bench(
            &home,
            events_path,
            &events,
            seed_events,
            iterations,
        );
    }

    let mut durations = Vec::with_capacity(iterations);
    for index in 0..iterations {
        let (tool, mut payload) = events[index % events.len()].clone();
        payload["message_id"] = json!(format!("bench-message-{index}"));
        payload["sequence"] = json!(index as i64);
        let started = Instant::now();
        ingest_hook_event(&home, tool, payload)?;
        durations.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    durations.sort_by(|left, right| left.total_cmp(right));
    let p95 = percentile(&durations, 0.95);
    let p99 = percentile(&durations, 0.99);

    Ok(json!({
        "iterations": iterations,
        "p95_ms": p95,
        "p99_ms": p99,
        "p95_under_50_ms": p95 < 50.0,
        "p99_under_250_ms": p99 < 250.0,
        "events": events_path,
        "home": home
    }))
}

fn run_large_session_ingest_bench(
    home: &Path,
    events_path: &PathBuf,
    events: &[(Tool, Value)],
    seed_events: usize,
    iterations: usize,
) -> nabu_core::Result<Value> {
    let tool = events[0].0;
    let session_id = "bench-large-session";
    seed_large_bench_session(home, tool, session_id, seed_events)?;

    let warm_payload = bench_user_payload(
        session_id,
        "bench-seed-message-0",
        0,
        "bench large-session seed marker 0",
    );
    let warm_report = ingest_hook_event(home, tool, warm_payload)?;
    if warm_report.appended {
        return Err(Error::Validation(
            "large-session bench warmup unexpectedly appended duplicate seed event".to_string(),
        ));
    }

    let mut new_durations = Vec::with_capacity(iterations);
    let mut duplicate_durations = Vec::with_capacity(iterations);
    for index in 0..iterations {
        let sequence = seed_events.saturating_add(index);
        let new_payload = bench_user_payload(
            session_id,
            &format!("bench-measured-message-{index}"),
            sequence,
            &format!("bench large-session measured marker {index}"),
        );
        let started = Instant::now();
        let new_report = ingest_hook_event(home, tool, new_payload)?;
        new_durations.push(started.elapsed().as_secs_f64() * 1000.0);
        if !new_report.appended {
            return Err(Error::Validation(
                "large-session bench measured new event was deduped".to_string(),
            ));
        }

        let duplicate_index = index % seed_events;
        let duplicate_payload = bench_user_payload(
            session_id,
            &format!("bench-seed-message-{duplicate_index}"),
            duplicate_index,
            &format!("bench large-session seed marker {duplicate_index}"),
        );
        let started = Instant::now();
        let duplicate_report = ingest_hook_event(home, tool, duplicate_payload)?;
        duplicate_durations.push(started.elapsed().as_secs_f64() * 1000.0);
        if duplicate_report.appended {
            return Err(Error::Validation(
                "large-session bench measured duplicate event appended".to_string(),
            ));
        }
    }

    new_durations.sort_by(|left, right| left.total_cmp(right));
    duplicate_durations.sort_by(|left, right| left.total_cmp(right));
    let new_p95 = percentile(&new_durations, 0.95);
    let new_p99 = percentile(&new_durations, 0.99);
    let duplicate_p95 = percentile(&duplicate_durations, 0.95);
    let duplicate_p99 = percentile(&duplicate_durations, 0.99);

    Ok(json!({
        "mode": "large_session",
        "iterations": iterations,
        "seed_events": seed_events,
        "events": events_path,
        "home": home,
        "new_event": {
            "p95_ms": new_p95,
            "p99_ms": new_p99,
            "p95_under_50_ms": new_p95 < 50.0,
            "p99_under_250_ms": new_p99 < 250.0
        },
        "duplicate_event": {
            "p95_ms": duplicate_p95,
            "p99_ms": duplicate_p99,
            "p95_under_50_ms": duplicate_p95 < 50.0,
            "p99_under_250_ms": duplicate_p99 < 250.0
        }
    }))
}

fn seed_large_bench_session(
    home: &Path,
    tool: Tool,
    session_id: &str,
    seed_events: usize,
) -> nabu_core::Result<()> {
    let raw_path = canonical_raw_path(home, tool, session_id);
    if let Some(parent) = raw_path.parent() {
        fs::create_dir_all(parent).map_err(|source| Error::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&raw_path)
        .map_err(|source| Error::Io {
            path: raw_path.clone(),
            source,
        })?;
    let mut raw_offset = file
        .metadata()
        .map_err(|source| Error::Io {
            path: raw_path.clone(),
            source,
        })?
        .len();
    for index in 0..seed_events {
        let payload = bench_user_payload(
            session_id,
            &format!("bench-seed-message-{index}"),
            index,
            &format!("bench large-session seed marker {index}"),
        );
        let source_event_id = format!("bench-seed-message-{index}");
        let dedupe_key = dedupe_key(DedupeParts {
            tool,
            session_id,
            canonical_type: CanonicalType::UserMessage,
            source_event_id: Some(&source_event_id),
            sequence: Some(index as i64),
            payload: &payload,
        })?;
        let envelope = EventEnvelope {
            schema_version: SCHEMA_VERSION,
            captured_at: "2026-06-18T00:00:00Z".to_string(),
            tool,
            tool_version: None,
            session_id: session_id.to_string(),
            filename_session_id: sanitize_session_id(session_id),
            turn_id: None,
            message_id: Some(source_event_id.clone()),
            project_root: Some("/tmp/nabu-bench".to_string()),
            cwd: Some("/tmp/nabu-bench".to_string()),
            source: Source::Hook,
            source_event_type: "UserPromptSubmit".to_string(),
            canonical_type: CanonicalType::UserMessage,
            source_event_id: Some(source_event_id),
            dedupe_key,
            sequence: Some(index as i64),
            raw_file: Some(raw_path.display().to_string()),
            raw_offset: Some(raw_offset as i64),
            payload,
            payload_ref: None,
        };
        envelope.validate()?;
        let line = serde_json::to_vec(&envelope)?;
        file.write_all(&line).map_err(|source| Error::Io {
            path: raw_path.clone(),
            source,
        })?;
        file.write_all(b"\n").map_err(|source| Error::Io {
            path: raw_path.clone(),
            source,
        })?;
        raw_offset += line.len() as u64 + 1;
    }
    Ok(())
}

fn bench_user_payload(session_id: &str, message_id: &str, sequence: usize, prompt: &str) -> Value {
    json!({
        "session_id": session_id,
        "hook_event_name": "UserPromptSubmit",
        "message_id": message_id,
        "sequence": sequence as i64,
        "cwd": "/tmp/nabu-bench",
        "project_root": "/tmp/nabu-bench",
        "prompt": prompt
    })
}

fn run_search_bench(
    home: &Path,
    query: &str,
    iterations: usize,
    limit: usize,
) -> nabu_core::Result<Value> {
    if query.trim().is_empty() {
        return Err(Error::Validation(
            "benchmark query must not be empty".to_string(),
        ));
    }
    let iterations = iterations.clamp(1, 10_000);
    let limit = limit.clamp(1, 50);
    let mut durations = Vec::with_capacity(iterations);
    let mut result_count = 0usize;
    for _ in 0..iterations {
        let started = Instant::now();
        let page = search_history_page(
            home,
            query,
            SearchOptions {
                limit,
                ..SearchOptions::default()
            },
        )?;
        durations.push(started.elapsed().as_secs_f64() * 1000.0);
        result_count = page.returned;
    }
    durations.sort_by(|left, right| left.total_cmp(right));
    let p95 = percentile(&durations, 0.95);
    let p99 = percentile(&durations, 0.99);

    Ok(json!({
        "iterations": iterations,
        "query": query,
        "limit": limit,
        "result_count": result_count,
        "p95_ms": p95,
        "p99_ms": p99,
        "p95_under_200_ms": p95 < 200.0,
        "home": home
    }))
}

fn follow_session_jsonl(home: &Path, tool: Tool, session_id: &str) -> nabu_core::Result<()> {
    let path = canonical_raw_path(home, tool, session_id);
    let mut position = 0u64;
    let mut line = String::new();
    loop {
        match File::open(&path) {
            Ok(mut file) => {
                file.seek(SeekFrom::Start(position))
                    .map_err(|source| Error::Io {
                        path: path.clone(),
                        source,
                    })?;
                let mut reader = BufReader::new(file);
                loop {
                    line.clear();
                    let bytes_read = reader.read_line(&mut line).map_err(|source| Error::Io {
                        path: path.clone(),
                        source,
                    })?;
                    if bytes_read == 0 {
                        break;
                    }
                    print!("{line}");
                    position += bytes_read as u64;
                }
                std::io::stdout().flush().map_err(|source| Error::Io {
                    path: PathBuf::from("<stdout>"),
                    source,
                })?;
            }
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => {}
            Err(source) => {
                return Err(Error::Io {
                    path: path.clone(),
                    source,
                })
            }
        }
        std::thread::sleep(Duration::from_secs(1));
    }
}

fn load_bench_events(events_path: &PathBuf) -> nabu_core::Result<Vec<(Tool, Value)>> {
    let file = File::open(events_path).map_err(|source| Error::Io {
        path: events_path.clone(),
        source,
    })?;
    let reader = BufReader::new(file);
    let mut events = Vec::new();

    for line in reader.lines() {
        let line = line.map_err(|source| Error::Io {
            path: events_path.clone(),
            source,
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let payload: Value = serde_json::from_str(&line)?;
        let tool = payload
            .get("tool")
            .and_then(Value::as_str)
            .map(Tool::from_str)
            .transpose()?
            .unwrap_or(Tool::Claude);
        events.push((tool, payload));
    }

    Ok(events)
}

fn percentile(sorted_values: &[f64], percentile: f64) -> f64 {
    let index = ((sorted_values.len() as f64 - 1.0) * percentile).ceil() as usize;
    sorted_values[index.min(sorted_values.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jsonc_edit::jsonc_to_json_value;
    use crate::testsupport::{file_mode, set_mode, EnvGuard, ENV_LOCK};
    use nabu_core::search_history;
    use tempfile::tempdir;

    #[test]
    fn parsed_command_controls_json_error_rendering_not_raw_argv() {
        let literal_query = Cli::try_parse_from(["nabu", "search", "--", "--json"]).unwrap();
        assert!(!literal_query.renders_errors_as_json());

        let json_search =
            Cli::try_parse_from(["nabu", "search", "needle", "--format", "json"]).unwrap();
        assert!(json_search.renders_errors_as_json());

        let json_doctor = Cli::try_parse_from(["nabu", "doctor", "--json"]).unwrap();
        assert!(json_doctor.renders_errors_as_json());

        let json_mcp = Cli::try_parse_from(["nabu", "mcp", "validate", "all", "--json"]).unwrap();
        assert!(json_mcp.renders_errors_as_json());
    }

    #[test]
    fn human_progress_renderer_includes_terminal_status_and_counts() {
        let rendered = ProgressEmitter::new(false).render(
            "index",
            "index",
            "completed",
            42,
            Some(42),
            "index pass completed",
        );

        assert_eq!(
            rendered,
            "progress operation=index phase=index status=completed items=42/42 message=index pass completed"
        );
    }

    #[test]
    fn json_progress_renderer_emits_ndjson_object_shape() {
        let rendered = ProgressEmitter::new(true).render(
            "embed.download",
            "model_acquisition",
            "failed",
            0,
            Some(1),
            "semantic backend unavailable in this build",
        );
        assert!(!rendered.contains('\n'));

        let payload: Value = serde_json::from_str(&rendered).unwrap();
        assert_eq!(payload["schema_version"], 1);
        assert_eq!(payload["operation"], "embed.download");
        assert_eq!(payload["phase"], "model_acquisition");
        assert_eq!(payload["status"], "failed");
        assert_eq!(payload["processed"], 0);
        assert_eq!(payload["total"], 1);
        assert_eq!(payload["unit"], "items");
        assert_eq!(
            payload["message"],
            "semantic backend unavailable in this build"
        );
        assert!(payload["timestamp"].as_str().unwrap().ends_with('Z'));
    }

    #[test]
    fn embedding_progress_renderer_includes_rate_eta_and_tuning() {
        let progress = EmbeddingIndexProgress {
            phase: "embedding".to_string(),
            status: "running".to_string(),
            embedded_units: 4096,
            total_units: 10_000,
            units_per_second: 512.25,
            eta_seconds: Some(12),
            batch_size: 64,
            write_chunk_size: 2048,
            intra_threads: 8,
        };

        let human = ProgressEmitter::new(false).render_embedding_index(&progress);
        assert_eq!(
            human,
            "progress operation=embed.index phase=embedding status=running units=4096/10000 rate=512.2/s eta=12s threads=8 batch=64 write_chunk=2048"
        );

        let rendered = ProgressEmitter::new(true).render_embedding_index(&progress);
        let payload: Value = serde_json::from_str(&rendered).unwrap();
        assert_eq!(payload["operation"], "embed.index");
        assert_eq!(payload["unit"], "units");
        assert_eq!(payload["processed"], 4096);
        assert_eq!(payload["total"], 10_000);
        assert_eq!(payload["units_per_second"], 512.25);
        assert_eq!(payload["eta_seconds"], 12);
        assert_eq!(payload["batch_size"], 64);
        assert_eq!(payload["write_chunk_size"], 2048);
        assert_eq!(payload["intra_threads"], 8);
    }

    #[test]
    fn embedding_plan_renderer_discloses_cpu_cost_and_no_embed_escape_hatch() {
        let progress = EmbeddingIndexProgress {
            phase: "embedding_plan".to_string(),
            status: "ready".to_string(),
            embedded_units: 0,
            total_units: 273_000,
            units_per_second: 0.0,
            eta_seconds: None,
            batch_size: 128,
            write_chunk_size: 2048,
            intra_threads: 10,
        };

        let human = ProgressEmitter::new(false).render_embedding_index(&progress);
        assert!(human.contains("phase=embedding_plan"));
        assert!(human.contains("273000 unembedded units"));
        assert!(human.contains("one-time CPU-intensive local pass"));
        assert!(human.contains("--no-embed"));

        let rendered = ProgressEmitter::new(true).render_embedding_index(&progress);
        let payload: Value = serde_json::from_str(&rendered).unwrap();
        assert_eq!(payload["operation"], "embed.index");
        assert_eq!(payload["phase"], "embedding_plan");
        assert_eq!(payload["status"], "ready");
        assert_eq!(payload["total"], 273_000);
        assert!(payload["message"].as_str().unwrap().contains("--no-embed"));
    }

    #[test]
    fn embedding_download_disclosure_renderer_lists_model_terms_and_footprint() {
        let disclosure = EmbeddingModelDisclosure {
            model_id: "embeddinggemma-300m-q4".to_string(),
            repository: "onnx-community/embeddinggemma-300m-ONNX".to_string(),
            cache_path: "/tmp/nabu/models/embeddinggemma-300m-q4".to_string(),
            total_files: 6,
            current_on_disk_bytes: 2048,
            model_present: false,
            license_summary: "Gemma Terms of Use: local model, explicit consent, no auto-download."
                .to_string(),
        };

        let human = ProgressEmitter::new(false).render_embedding_download_disclosure(&disclosure);
        assert_eq!(human.len(), 3);
        assert!(human[0].contains("operation=embed.download"));
        assert!(human[0].contains("phase=model_disclosure"));
        assert!(human[0].contains("embeddinggemma-300m-q4"));
        assert!(human[1].contains("Gemma Terms of Use"));
        assert!(human[2].contains("measured local footprint"));
        assert!(human[2].contains("2.0 KB"));

        let rendered = ProgressEmitter::new(true).render_embedding_download_disclosure(&disclosure);
        let payload: Value = serde_json::from_str(&rendered[0]).unwrap();
        assert_eq!(payload["operation"], "embed.download");
        assert_eq!(payload["phase"], "model_disclosure");
        assert_eq!(payload["status"], "info");
        assert_eq!(payload["total"], 6);
    }

    #[test]
    fn search_benchmark_reports_latency_gate() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();
        ingest_hook_event(
            &home,
            Tool::Claude,
            json!({
                "session_id": "bench-search-session",
                "hook_event_name": "UserPromptSubmit",
                "message_id": "bench-search-1",
                "cwd": "/tmp/nabu-fixture",
                "project_root": "/tmp/nabu-fixture",
                "prompt": "bench search marker"
            }),
        )
        .unwrap();
        index_once(&home).unwrap();

        let report = run_search_bench(&home, "bench search marker", 2, 10).unwrap();

        assert_eq!(report["iterations"], 2);
        assert_eq!(report["result_count"], 1);
        assert!(report["p95_ms"].is_number());
        assert!(report["p95_under_200_ms"].is_boolean());
    }

    #[test]
    fn large_session_ingest_benchmark_exercises_seeded_append_paths() {
        let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let report = run_ingest_bench(&repo.join("fixtures/bench/events.jsonl"), 10_000, 3)
            .expect("large-session ingest bench should run");

        assert_eq!(report["mode"], "large_session");
        assert_eq!(report["seed_events"], 10_000);
        assert_eq!(report["iterations"], 3);
        assert!(report["new_event"]["p95_ms"].is_number());
        assert!(report["new_event"]["p99_ms"].is_number());
        assert!(report["duplicate_event"]["p95_ms"].is_number());
        assert!(report["duplicate_event"]["p99_ms"].is_number());
        assert!(report["new_event"]["p95_under_50_ms"].is_boolean());
        assert!(report["duplicate_event"]["p95_under_50_ms"].is_boolean());
    }

    #[test]
    fn opencode_server_api_backfill_fetches_configured_session_messages() {
        let _guard = ENV_LOCK.lock().unwrap();
        let temp = tempdir().unwrap();
        let harness_home = temp.path().join("harness-home");
        let opencode_root = temp.path().join("opencode");
        let session_id = "ddddddd4-dddd-4ddd-8ddd-ddddddddddd4";
        init_home(&harness_home).unwrap();
        fs::create_dir_all(opencode_root.join("storage/message").join(session_id)).unwrap();
        ingest_hook_event(
            &harness_home,
            Tool::Opencode,
            json!({
                "session_id": session_id,
                "hook_event_name": "message.updated",
                "id": "opencode-api-shared-message",
                "role": "assistant",
                "text": "opencode configured server shared marker"
            }),
        )
        .unwrap();

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0u8; 2048];
            let bytes = stream.read(&mut request).unwrap();
            let request = String::from_utf8_lossy(&request[..bytes]);
            assert!(request.contains(&format!("GET /session/{session_id}/message ")));
            let body = serde_json::to_string(&json!([
                {
                    "id": "opencode-api-shared-message",
                    "sessionID": session_id,
                    "role": "assistant",
                    "text": "opencode configured server shared marker"
                },
                {
                    "id": "opencode-api-gap-message",
                    "sessionID": session_id,
                    "role": "assistant",
                    "parts": [
                        {
                            "id": "opencode-api-gap-part",
                            "type": "text",
                            "text": "opencode configured server api marker"
                        }
                    ]
                }
            ]))
            .unwrap();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).unwrap();
        });

        fs::write(
            harness_home.join("config.toml"),
            format!(
                "schema_version = 1\n\n[opencode]\nserver_url = \"http://{}\"\n",
                address
            ),
        )
        .unwrap();
        let env_guard = EnvGuard::set([
            ("NABU_OPENCODE_URL", std::ffi::OsStr::new("")),
            ("TUPSHARRUM_OPENCODE_URL", std::ffi::OsStr::new("")),
        ]);

        let report =
            backfill_opencode_server_api_if_configured(&harness_home, &opencode_root, None)
                .unwrap();
        server.join().unwrap();

        assert_eq!(report.source_files, 1);
        assert_eq!(report.appended_events, 1);
        assert!(!harness_home.join("spool/opencode-api").exists());
        index_once(&harness_home).unwrap();
        let results =
            search_history(&harness_home, "opencode configured server api marker", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].session_id, session_id);
        drop(env_guard);
    }

    #[test]
    fn opencode_server_api_backfill_skips_without_configured_url() {
        let _guard = ENV_LOCK.lock().unwrap();
        let temp = tempdir().unwrap();
        let harness_home = temp.path().join("harness-home");
        let opencode_root = temp.path().join("opencode");
        let session_id = "eeeeeee5-eeee-4eee-8eee-eeeeeeeeeee5";
        init_home(&harness_home).unwrap();
        fs::create_dir_all(opencode_root.join("storage/message").join(session_id)).unwrap();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let env_guard = EnvGuard::set([
            ("NABU_OPENCODE_URL", std::ffi::OsStr::new("")),
            ("TUPSHARRUM_OPENCODE_URL", std::ffi::OsStr::new("")),
        ]);

        let report =
            backfill_opencode_server_api_if_configured(&harness_home, &opencode_root, None)
                .unwrap();

        assert_eq!(report.source_files, 0);
        assert_eq!(report.appended_events, 0);
        assert!(matches!(
            listener.accept(),
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock
        ));
        drop(env_guard);
    }

    #[test]
    fn opencode_server_api_backfill_logs_and_continues_on_server_error() {
        let _guard = ENV_LOCK.lock().unwrap();
        let temp = tempdir().unwrap();
        let harness_home = temp.path().join("harness-home");
        let opencode_root = temp.path().join("opencode");
        let session_id = "fffffff6-ffff-4fff-8fff-fffffffffff6";
        init_home(&harness_home).unwrap();
        fs::create_dir_all(opencode_root.join("storage/message").join(session_id)).unwrap();

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let response =
                "HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
            stream.write_all(response.as_bytes()).unwrap();
        });
        fs::write(
            harness_home.join("config.toml"),
            format!(
                "schema_version = 1\n\n[opencode]\nserver_url = \"http://{}\"\n",
                address
            ),
        )
        .unwrap();
        let env_guard = EnvGuard::set([
            ("NABU_OPENCODE_URL", std::ffi::OsStr::new("")),
            ("TUPSHARRUM_OPENCODE_URL", std::ffi::OsStr::new("")),
        ]);

        let report =
            backfill_opencode_server_api_if_configured(&harness_home, &opencode_root, None)
                .unwrap();
        server.join().unwrap();

        assert_eq!(report.source_files, 0);
        assert_eq!(report.appended_events, 0);
        assert!(!canonical_raw_path(&harness_home, Tool::Opencode, session_id).exists());
        drop(env_guard);
    }

    #[test]
    fn doctor_json_reports_codex_and_opencode_parity_shape() {
        let _guard = ENV_LOCK.lock().unwrap();
        let temp = tempdir().unwrap();
        let harness_home = temp.path().join("harness-home");
        let fake_home = temp.path().join("home");
        let fake_bin = temp.path().join("bin");
        let codex_home = temp.path().join("codex");
        let opencode_config = temp.path().join("opencode-config");
        init_home(&harness_home).unwrap();
        fs::create_dir_all(&fake_home).unwrap();
        fs::create_dir_all(&fake_bin).unwrap();
        fs::create_dir_all(&codex_home).unwrap();
        fs::create_dir_all(&opencode_config).unwrap();
        fs::write(
            harness_home.join("config.toml"),
            "schema_version = 1\n\n[opencode]\nserver_url = \"http://127.0.0.1:4096\"\n",
        )
        .unwrap();
        let env_guard = EnvGuard::set([
            ("HOME", fake_home.as_os_str()),
            ("CODEX_HOME", codex_home.as_os_str()),
            ("OPENCODE_CONFIG_DIR", opencode_config.as_os_str()),
            ("PATH", fake_bin.as_os_str()),
            ("NABU_OPENCODE_URL", std::ffi::OsStr::new("")),
            ("TUPSHARRUM_OPENCODE_URL", std::ffi::OsStr::new("")),
        ]);
        install_codex(&harness_home, false).unwrap();
        install_opencode(&harness_home, false).unwrap();
        ingest_hook_event(
            &harness_home,
            Tool::Codex,
            json!({
                "session_id": "doctor-codex-session",
                "hook_event_name": "UserPromptSubmit",
                "message_id": "doctor-codex-1",
                "prompt": "doctor codex marker"
            }),
        )
        .unwrap();
        ingest_hook_event(
            &harness_home,
            Tool::Opencode,
            json!({
                "session_id": "doctor-opencode-session",
                "hook_event_name": "message.updated",
                "id": "doctor-opencode-1",
                "text": "doctor opencode marker"
            }),
        )
        .unwrap();
        index_once(&harness_home).unwrap();

        let codex = doctor_json_data(&harness_home, DoctorTool::Codex, false).unwrap();
        let opencode = doctor_json_data(&harness_home, DoctorTool::Opencode, false).unwrap();

        assert_eq!(
            codex.pointer("/tools/codex/hooks_installed"),
            Some(&json!(true))
        );
        assert_eq!(
            codex.pointer("/tools/codex/status"),
            Some(&json!("not_applicable"))
        );
        assert!(codex.pointer("/tools/codex/hooks_path").is_some());
        assert!(codex
            .pointer("/tools/codex/latest_captured_event/session_id")
            .is_some());
        assert_eq!(
            opencode.pointer("/tools/opencode/plugin_installed"),
            Some(&json!(true))
        );
        assert_eq!(
            opencode.pointer("/tools/opencode/status"),
            Some(&json!("not_applicable"))
        );
        assert_eq!(
            opencode.pointer("/tools/opencode/reconciliation_enabled"),
            Some(&json!(true))
        );
        assert!(opencode
            .pointer("/tools/opencode/latest_captured_event/session_id")
            .is_some());
        drop(env_guard);
    }

    #[test]
    fn default_backfill_scans_native_roots_for_all_tools() {
        let _guard = ENV_LOCK.lock().unwrap();
        let temp = tempdir().unwrap();
        let native = temp.path().join("native");
        let codex_home = native.join("codex-home");
        let claude_config = native.join("claude-config");
        let fake_home = native.join("home");
        fs::create_dir_all(codex_home.join("sessions")).unwrap();
        fs::create_dir_all(claude_config.join("projects")).unwrap();
        fs::create_dir_all(fake_home.join(".local/share/opencode/project")).unwrap();

        let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        fs::copy(
            repo.join("fixtures/codex/sessions/session_fixture.jsonl"),
            codex_home.join("sessions/session_fixture.jsonl"),
        )
        .unwrap();
        fs::copy(
            repo.join("fixtures/claude-code/transcript.jsonl"),
            claude_config.join("projects/transcript.jsonl"),
        )
        .unwrap();
        fs::copy(
            repo.join("fixtures/opencode/server_session_messages.json"),
            fake_home.join(".local/share/opencode/project/server_session_messages.json"),
        )
        .unwrap();

        let env_guard = EnvGuard::set([
            ("CODEX_HOME", codex_home.as_os_str()),
            ("CLAUDE_CONFIG_DIR", claude_config.as_os_str()),
            ("HOME", fake_home.as_os_str()),
        ]);
        let harness_home = temp.path().join("harness-home");
        init_home(&harness_home).unwrap();

        let report = run_backfill_command(
            &harness_home,
            BackfillTool::All,
            None,
            None,
            ProgressEmitter::new(false),
        )
        .unwrap();
        assert_eq!(report.source_files, 3);
        assert_eq!(report.appended_events, 4);
        assert_eq!(report.checkpoint_files, 3);

        index_once(&harness_home).unwrap();
        assert_eq!(
            search_history(&harness_home, "codex fixture marker", 1).unwrap()[0].tool,
            Tool::Codex
        );
        assert_eq!(
            search_history(&harness_home, "fixture unique phrase", 1).unwrap()[0].tool,
            Tool::Claude
        );
        assert_eq!(
            search_history(&harness_home, "opencode fixture marker", 1).unwrap()[0].tool,
            Tool::Opencode
        );
        drop(env_guard);
    }

    #[test]
    fn mcp_install_removes_legacy_tupsharrum_entry_no_orphans() {
        // Installing the current server over a config that still holds the
        // pre-rename `tupsharrum` entry must drop it — an upgrade leaves no
        // server pointing at a removed binary. (Regression: install previously
        // only added `nabu` and, when already present, wrote nothing.)
        let opencode = add_opencode_mcp(json!({
            "mcp": { "tupsharrum": { "command": ["tupsharrum", "mcp", "serve"] } }
        }));
        let mcp = opencode.get("mcp").unwrap().as_object().unwrap();
        assert!(mcp.contains_key("nabu"));
        assert!(!mcp.contains_key("tupsharrum"));

        let claude = add_claude_mcp(json!({
            "mcpServers": { "tupsharrum": { "command": "tupsharrum" } }
        }));
        let servers = claude.get("mcpServers").unwrap().as_object().unwrap();
        assert!(servers.contains_key("nabu"));
        assert!(!servers.contains_key("tupsharrum"));

        let codex = add_codex_mcp_block(
            "[mcp_servers.tupsharrum]\ncommand = \"tupsharrum\"\nenabled = true\n",
        );
        assert!(codex.contains("[mcp_servers.nabu]"));
        assert!(!codex.contains("tupsharrum"));
    }

    #[test]
    fn mcp_install_uninstall_preserves_configs_and_records_backups() {
        let _guard = ENV_LOCK.lock().unwrap();
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        let codex_home = temp.path().join("codex");
        let opencode_config = temp.path().join("opencode");
        let raven_home = temp.path().join("raven");
        fs::create_dir_all(&home).unwrap();
        fs::create_dir_all(&codex_home).unwrap();
        fs::create_dir_all(&opencode_config).unwrap();
        init_home(&raven_home).unwrap();

        let codex_config = codex_home.join("config.toml");
        let claude_config = home.join(".claude.json");
        let opencode_json = opencode_config.join("opencode.json");
        fs::write(&codex_config, "[model]\nname = \"fixture\"\n").unwrap();
        fs::write(&claude_config, "{\"theme\":\"dark\"}").unwrap();
        fs::write(&opencode_json, "{\"theme\":\"dark\"}").unwrap();
        set_mode(&codex_config, 0o640);
        set_mode(&claude_config, 0o640);
        set_mode(&opencode_json, 0o644);

        let env_guard = EnvGuard::set([
            ("HOME", home.as_os_str()),
            ("CODEX_HOME", codex_home.as_os_str()),
            ("OPENCODE_CONFIG_DIR", opencode_config.as_os_str()),
            ("PATH", std::ffi::OsStr::new("/usr/bin:/bin")),
        ]);

        let before_codex = fs::read_to_string(&codex_config).unwrap();
        let before_claude = fs::read_to_string(&claude_config).unwrap();
        let before_opencode = fs::read_to_string(&opencode_json).unwrap();
        let dry_run_reports =
            mcp_apply_all(&raven_home, AgentTool::All, McpConfigAction::Install, true).unwrap();
        assert_eq!(dry_run_reports.len(), 3);
        assert!(dry_run_reports.iter().all(|report| report.dry_run));
        assert_eq!(fs::read_to_string(&codex_config).unwrap(), before_codex);
        assert_eq!(fs::read_to_string(&claude_config).unwrap(), before_claude);
        assert_eq!(fs::read_to_string(&opencode_json).unwrap(), before_opencode);
        assert!(!raven_home.join("backups/manifest.jsonl").exists());

        let install_reports =
            mcp_apply_all(&raven_home, AgentTool::All, McpConfigAction::Install, false).unwrap();
        assert!(install_reports.iter().all(|report| report.changed));
        assert!(fs::read_to_string(&codex_config)
            .unwrap()
            .contains("[mcp_servers.nabu]"));
        assert!(fs::read_to_string(&codex_config)
            .unwrap()
            .contains("[model]"));
        let claude: Value =
            serde_json::from_str(&fs::read_to_string(&claude_config).unwrap()).unwrap();
        assert_eq!(claude["theme"], "dark");
        assert!(claude.pointer("/mcpServers/nabu").is_some());
        let opencode: Value =
            serde_json::from_str(&fs::read_to_string(&opencode_json).unwrap()).unwrap();
        assert_eq!(opencode["theme"], "dark");
        assert!(opencode.pointer("/mcp/nabu").is_some());
        assert_eq!(file_mode(&codex_config), 0o640);
        assert_eq!(file_mode(&claude_config), 0o640);
        assert_eq!(file_mode(&opencode_json), 0o644);

        let manifest_path = raven_home.join("backups/manifest.jsonl");
        let install_manifest = fs::read_to_string(&manifest_path).unwrap();
        assert_eq!(install_manifest.lines().count(), 3);
        for line in install_manifest.lines() {
            let record: Value = serde_json::from_str(line).unwrap();
            assert_eq!(record["operation"], "mcp-install");
            let backup_path = PathBuf::from(record["backup_path"].as_str().unwrap());
            assert!(backup_path.is_file());
            assert_eq!(file_mode(&backup_path), 0o600);
            assert!(backup_path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .contains(".nabu-backup."));
        }

        let idempotent_reports =
            mcp_apply_all(&raven_home, AgentTool::All, McpConfigAction::Install, false).unwrap();
        assert!(idempotent_reports.iter().all(|report| !report.changed));
        assert_eq!(
            fs::read_to_string(&manifest_path).unwrap().lines().count(),
            3
        );

        let uninstall_reports = mcp_apply_all(
            &raven_home,
            AgentTool::All,
            McpConfigAction::Uninstall,
            false,
        )
        .unwrap();
        assert!(uninstall_reports.iter().all(|report| report.changed));
        assert!(fs::read_to_string(&codex_config)
            .unwrap()
            .contains("[model]"));
        assert!(!fs::read_to_string(&codex_config)
            .unwrap()
            .contains("[mcp_servers.nabu]"));
        let claude: Value =
            serde_json::from_str(&fs::read_to_string(&claude_config).unwrap()).unwrap();
        assert_eq!(claude["theme"], "dark");
        assert!(claude.pointer("/mcpServers/nabu").is_none());
        let opencode: Value =
            serde_json::from_str(&fs::read_to_string(&opencode_json).unwrap()).unwrap();
        assert_eq!(opencode["theme"], "dark");
        assert!(opencode.pointer("/mcp/nabu").is_none());
        assert_eq!(
            fs::read_to_string(&manifest_path).unwrap().lines().count(),
            6
        );
        drop(env_guard);
    }

    #[test]
    fn opencode_mcp_install_preserves_jsonc_comments_without_reformatting() {
        let _guard = ENV_LOCK.lock().unwrap();
        let temp = tempdir().unwrap();
        let harness_home = temp.path().join("harness-home");
        let opencode_config = temp.path().join("opencode");
        fs::create_dir_all(&opencode_config).unwrap();
        init_home(&harness_home).unwrap();

        let opencode_json = opencode_config.join("opencode.json");
        fs::write(
            &opencode_json,
            "{\n  // keep this comment\n  \"theme\": \"dark\"\n}\n",
        )
        .unwrap();
        let env_guard = EnvGuard::set([("OPENCODE_CONFIG_DIR", opencode_config.as_os_str())]);

        let report = mcp_apply_opencode(&harness_home, McpConfigAction::Install, false).unwrap();
        assert!(report.changed);
        let after = fs::read_to_string(&opencode_json).unwrap();
        assert!(after.contains("// keep this comment"));
        assert!(after.contains("\"theme\": \"dark\""));
        assert!(after.contains("\"mcp\""));
        assert!(after.contains("\"nabu\""));
        assert!(after.contains("\"command\""));
        assert!(jsonc_to_json_value(&after).pointer("/mcp/nabu").is_some());
        assert!(opencode_mcp_entry_installed());

        let idempotent =
            mcp_apply_opencode(&harness_home, McpConfigAction::Install, false).unwrap();
        assert!(!idempotent.changed);
        assert_eq!(fs::read_to_string(&opencode_json).unwrap(), after);
        drop(env_guard);
    }

    #[test]
    fn opencode_mcp_install_updates_existing_jsonc_mcp_without_reformatting() {
        let _guard = ENV_LOCK.lock().unwrap();
        let temp = tempdir().unwrap();
        let harness_home = temp.path().join("harness-home");
        let opencode_config = temp.path().join("opencode");
        fs::create_dir_all(&opencode_config).unwrap();
        init_home(&harness_home).unwrap();

        let opencode_json = opencode_config.join("opencode.json");
        fs::write(
            &opencode_json,
            "{\n  // keep this comment\n  \"theme\": \"dark\",\n  \"mcp\": {\n    // keep other server\n    \"other\": { \"type\": \"local\", \"command\": [\"other\"] }\n  }\n}\n",
        )
        .unwrap();
        let env_guard = EnvGuard::set([("OPENCODE_CONFIG_DIR", opencode_config.as_os_str())]);

        let report = mcp_apply_opencode(&harness_home, McpConfigAction::Install, false).unwrap();
        assert!(report.changed);
        let after_install = fs::read_to_string(&opencode_json).unwrap();
        assert!(after_install.contains("// keep this comment"));
        assert!(after_install.contains("// keep other server"));
        assert!(after_install.contains("\"other\": { \"type\": \"local\""));
        assert!(after_install.contains("\"nabu\""));
        assert!(jsonc_to_json_value(&after_install)
            .pointer("/mcp/nabu")
            .is_some());
        assert!(opencode_mcp_entry_installed());

        let uninstall =
            mcp_apply_opencode(&harness_home, McpConfigAction::Uninstall, false).unwrap();
        assert!(uninstall.changed);
        let after_uninstall = fs::read_to_string(&opencode_json).unwrap();
        assert!(after_uninstall.contains("// keep this comment"));
        assert!(after_uninstall.contains("// keep other server"));
        assert!(after_uninstall.contains("\"other\": { \"type\": \"local\""));
        assert!(!after_uninstall.contains("\"nabu\""));
        assert!(jsonc_to_json_value(&after_uninstall)
            .pointer("/mcp/nabu")
            .is_none());
        assert!(!opencode_mcp_entry_installed());
        drop(env_guard);
    }

    #[test]
    fn opencode_mcp_install_preserves_same_line_comments_crlf_and_bom() {
        let _guard = ENV_LOCK.lock().unwrap();
        let temp = tempdir().unwrap();
        let harness_home = temp.path().join("harness-home");
        let opencode_config = temp.path().join("opencode");
        fs::create_dir_all(&opencode_config).unwrap();
        init_home(&harness_home).unwrap();

        let opencode_json = opencode_config.join("opencode.json");
        let before = "\u{feff}{\r\n  \"theme\": \"dark\",\r\n  \"mcp\": {\r\n    \"other\": { \"type\": \"local\", \"command\": [\"other\"] } // keep same-line comment\r\n  }\r\n}\r\n";
        fs::write(&opencode_json, before).unwrap();
        let env_guard = EnvGuard::set([("OPENCODE_CONFIG_DIR", opencode_config.as_os_str())]);

        let report = mcp_apply_opencode(&harness_home, McpConfigAction::Install, false).unwrap();
        assert!(report.changed);
        let after = fs::read_to_string(&opencode_json).unwrap();
        assert!(after.starts_with('\u{feff}'));
        assert!(after.contains("}, // keep same-line comment"));
        assert!(!after.replace("\r\n", "").contains('\n'));
        let parsed = jsonc_to_json_value(&after);
        assert_eq!(parsed["theme"], "dark");
        assert!(parsed.pointer("/mcp/other").is_some());
        assert!(parsed.pointer("/mcp/nabu").is_some());
        drop(env_guard);
    }

    #[test]
    fn opencode_mcp_install_rejects_duplicate_jsonc_mcp_keys_without_rewriting() {
        let _guard = ENV_LOCK.lock().unwrap();
        let temp = tempdir().unwrap();
        let harness_home = temp.path().join("harness-home");
        let opencode_config = temp.path().join("opencode");
        fs::create_dir_all(&opencode_config).unwrap();
        init_home(&harness_home).unwrap();

        let opencode_json = opencode_config.join("opencode.json");
        let before = "{\n  \"mcp\": {},\n  // ambiguous duplicate\n  \"mcp\": {}\n}\n";
        fs::write(&opencode_json, before).unwrap();
        let env_guard = EnvGuard::set([("OPENCODE_CONFIG_DIR", opencode_config.as_os_str())]);

        let error = mcp_apply_opencode(&harness_home, McpConfigAction::Install, false).unwrap_err();
        assert!(matches!(error, Error::Validation(_)));
        assert_eq!(fs::read_to_string(&opencode_json).unwrap(), before);
        drop(env_guard);
    }

    #[cfg(unix)]
    #[test]
    fn wizard_full_consent_reaches_setup_end_state_and_reruns_idempotent() {
        use nabu_core::search_history;
        use std::os::unix::fs::PermissionsExt;

        let _guard = ENV_LOCK.lock().unwrap();
        let temp = tempdir().unwrap();
        let harness_home = temp.path().join("harness-home");
        let fake_home = temp.path().join("home");
        let fake_bin = temp.path().join("bin");
        let codex_home = temp.path().join("codex");
        let claude_config = temp.path().join("claude");
        let opencode_config = temp.path().join("opencode");
        for dir in [
            &fake_home,
            &fake_bin,
            &codex_home,
            &claude_config,
            &opencode_config,
        ] {
            fs::create_dir_all(dir).unwrap();
        }
        // Native backfill roots must exist; seed Claude with a real transcript.
        fs::create_dir_all(codex_home.join("sessions")).unwrap();
        fs::create_dir_all(codex_home.join("archived_sessions")).unwrap();
        fs::create_dir_all(claude_config.join("projects")).unwrap();
        fs::create_dir_all(fake_home.join(".local/share/opencode")).unwrap();
        let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        fs::copy(
            repo.join("fixtures/claude-code/transcript.jsonl"),
            claude_config.join("projects/transcript.jsonl"),
        )
        .unwrap();

        // Fake executables so PATH detection finds all three tools. Only
        // `claude` is ever executed (its native MCP CLI): make it a no-op.
        for name in ["codex", "claude", "opencode"] {
            let path = fake_bin.join(name);
            fs::write(&path, "#!/bin/sh\nexit 0\n").unwrap();
            fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
        }

        // Pre-existing agent configs so MCP install must back them up.
        let codex_toml = codex_home.join("config.toml");
        let opencode_json = opencode_config.join("opencode.json");
        fs::write(&codex_toml, "[model]\nname = \"fixture\"\n").unwrap();
        fs::write(&opencode_json, "{\"theme\":\"dark\"}").unwrap();

        init_home(&harness_home).unwrap();

        let env_guard = EnvGuard::set([
            ("HOME", fake_home.as_os_str()),
            ("CODEX_HOME", codex_home.as_os_str()),
            ("CLAUDE_CONFIG_DIR", claude_config.as_os_str()),
            ("OPENCODE_CONFIG_DIR", opencode_config.as_os_str()),
            ("PATH", fake_bin.as_os_str()),
            ("NABU_OPENCODE_URL", std::ffi::OsStr::new("")),
            ("TUPSHARRUM_OPENCODE_URL", std::ffi::OsStr::new("")),
        ]);

        // First run: full consent through Get started, then Quit. Capture,
        // backfill, and connect are checklists that default to all tools, so the
        // multi-select queue is left empty (= accept all); only menu pick and
        // Quit are scripted as selects.
        let mut prompter = wizard::ScriptedPrompter::new()
            .selects([0usize, 6usize])
            .confirms(std::iter::repeat_n(true, 12));
        let mut actions = wizard::LiveActions;
        wizard::run(&mut prompter, &mut actions, &harness_home).unwrap();

        // Same end state as init + install all + backfill + mcp install.
        let codex_hooks: Value =
            serde_json::from_str(&fs::read_to_string(codex_home.join("hooks.json")).unwrap())
                .unwrap();
        assert!(codex_hooks
            .to_string()
            .contains("nabu ingest hook --tool codex"));
        let claude_settings: Value =
            serde_json::from_str(&fs::read_to_string(claude_config.join("settings.json")).unwrap())
                .unwrap();
        assert!(claude_settings
            .to_string()
            .contains("nabu ingest hook --tool claude"));
        assert!(opencode_config.join("plugins/harness-history.ts").is_file());
        let codex_toml_after = fs::read_to_string(&codex_toml).unwrap();
        assert!(codex_toml_after.contains("[mcp_servers.nabu]"));
        assert!(codex_toml_after.contains("[model]")); // unrelated config preserved
        let opencode_after: Value =
            serde_json::from_str(&fs::read_to_string(&opencode_json).unwrap()).unwrap();
        assert!(opencode_after.pointer("/mcp/nabu").is_some());
        assert_eq!(opencode_after["theme"], "dark"); // unrelated config preserved

        // Backfill imported the seeded Claude transcript.
        index_once(&harness_home).unwrap();
        assert!(!search_history(&harness_home, "fixture unique phrase", 1)
            .unwrap()
            .is_empty());

        // Every applied change to a pre-existing file has a timestamped backup.
        let manifest_path = harness_home.join("backups/manifest.jsonl");
        let backups_after_first = fs::read_to_string(&manifest_path).unwrap().lines().count();
        assert!(
            backups_after_first >= 2,
            "expected codex + opencode MCP backups, got {backups_after_first}"
        );

        // Hooks are not duplicated: exactly one nabu entry per codex event.
        for (_event, entries) in codex_hooks
            .pointer("/hooks")
            .and_then(Value::as_object)
            .unwrap()
        {
            let count = entries
                .as_array()
                .unwrap()
                .iter()
                .filter(|entry| {
                    entry
                        .get("command")
                        .and_then(Value::as_str)
                        .map(|command| command.contains("nabu"))
                        .unwrap_or(false)
                })
                .count();
            assert_eq!(count, 1);
        }

        // Re-run: idempotent — configured tools are not reinstalled, no new
        // agent-config backups, hooks unchanged.
        let mut prompter2 = wizard::ScriptedPrompter::new()
            .selects([0usize, 6usize])
            .confirms(std::iter::repeat_n(true, 12));
        let mut actions2 = wizard::LiveActions;
        wizard::run(&mut prompter2, &mut actions2, &harness_home).unwrap();

        let backups_after_second = fs::read_to_string(&manifest_path).unwrap().lines().count();
        assert_eq!(
            backups_after_second, backups_after_first,
            "re-run must not back up unchanged configs"
        );
        let codex_hooks_rerun: Value =
            serde_json::from_str(&fs::read_to_string(codex_home.join("hooks.json")).unwrap())
                .unwrap();
        assert_eq!(
            codex_hooks_rerun, codex_hooks,
            "re-run must not modify codex hooks"
        );

        drop(env_guard);
    }
}
