mod backfill;
mod backup;
mod bench;
mod jsonc_edit;
mod mcp_config;
mod opencode_http;
mod paths;
mod progress;
mod wizard;

#[cfg(test)]
mod testsupport;

use crate::backfill::{run_backfill_command, run_backfill_dry_run_command};
use crate::bench::{run_ingest_bench, run_search_bench};
use crate::mcp_config::{
    claude_mcp_entry_installed, codex_mcp_entry_installed, mcp_apply_all, mcp_validate_all,
    opencode_mcp_entry_installed,
};
use crate::progress::ProgressEmitter;
use clap::{Parser, Subcommand, ValueEnum};
use nabu_adapters::{
    claude_status, codex_status, install_claude, install_codex, install_opencode, opencode_status,
    uninstall_claude, uninstall_codex, uninstall_opencode, ConfigChangeReport,
};
#[cfg(test)]
use nabu_core::index_once;
use nabu_core::{
    canonical_raw_path, doctor_with_options, download_embedding_model_with_progress,
    embedding_model_disclosure, embedding_model_status, export_session_jsonl_with_options,
    export_session_markdown_with_options, index_once_with_options_and_progress, ingest_file,
    ingest_hook_event, init_home, latest_event, prune_embedding_cache, purge_all, purge_before,
    purge_session, resolve_home, search_history_page, Corroboration, Error, IndexOptions,
    PurgeAction, PurgeAllOptions, PurgeAllReport, PurgeTier, SearchMode, SearchOptions,
    SessionOptions, Source, Tool,
};
use serde_json::{json, Value};
use std::fs::File;
use std::io::{BufRead, BufReader, IsTerminal, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Duration;

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

pub(crate) struct ByteSize(u64);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backfill::backfill_opencode_server_api_if_configured;
    use crate::jsonc_edit::jsonc_to_json_value;
    use crate::mcp_config::{
        add_claude_mcp, add_codex_mcp_block, add_opencode_mcp, mcp_apply_opencode,
    };
    use crate::testsupport::{file_mode, set_mode, EnvGuard, ENV_LOCK};
    use nabu_core::{search_history, EmbeddingIndexProgress, EmbeddingModelDisclosure};
    use std::fs;
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
