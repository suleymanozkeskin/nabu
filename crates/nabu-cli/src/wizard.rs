//! `nabu wizard` — guided first-run and management front end.
//!
//! The wizard is a state machine over a [`Prompter`] (all interactive IO) and a
//! [`WizardActions`] (every library call). It owns detection, consent, and
//! sequencing only: every config mutation is delegated to the existing
//! install/uninstall/backfill/mcp functions, which already preview a diff and
//! write a timestamped backup. The wizard adds no new config write path and
//! never enables redaction.
//!
//! Separating logic (the state machine) from IO (the `Prompter`) is what makes
//! every acceptance criterion testable without a TTY: tests drive the machine
//! with a scripted `Prompter` and spy `WizardActions`, asserting which library
//! functions ran with which arguments.

use std::io::IsTerminal;
use std::path::{Path, PathBuf};

use nabu_adapters::{
    claude_status, codex_status, install_claude, install_codex, install_opencode, opencode_status,
    uninstall_claude, uninstall_codex, uninstall_opencode, ConfigChangeReport,
};
use nabu_core::{
    doctor_with_options, embedding_model_status, init_home, opencode_server_url,
    search_history_page, BackfillDryRunReport, BackfillReport, DoctorReport, Error, Result,
    SearchOptions, Tool,
};

use crate::{
    claude_mcp_entry_installed, codex_mcp_entry_installed, mcp_apply_one,
    opencode_mcp_entry_installed, run_backfill_command, run_backfill_dry_run_command, AgentTool,
    BackfillTool, McpConfigAction, ProgressEmitter,
};

/// The top-level menu, in display order. Indices are stable so tests can script
/// menu navigation against named constants.
const TOP_MENU: [&str; 7] = [
    "Get started",
    "Manage integrations",
    "Backfill history",
    "Settings",
    "Health check",
    "Connect agents (MCP)",
    "Quit",
];
const TOP_GET_STARTED: usize = 0;
const TOP_MANAGE: usize = 1;
const TOP_BACKFILL: usize = 2;
const TOP_SETTINGS: usize = 3;
const TOP_HEALTH: usize = 4;
const TOP_CONNECT: usize = 5;
const TOP_QUIT: usize = 6;

const NON_TTY_MESSAGE: &str = "nabu wizard needs an interactive terminal.\n\
Run the explicit commands instead (preview any step with --dry-run):\n\
  nabu init\n\
  nabu install all\n\
  nabu backfill --tool all\n\
  nabu mcp install all\n\
  nabu doctor";

const SAMPLE_QUERY: &str = "what did I change in the database schema";
const SAMPLE_SEARCH_COMMAND: &str = "nabu search \"what did I change in the database schema\"";

// ---------------------------------------------------------------------------
// Prompter: all interactive IO lives behind this trait.
// ---------------------------------------------------------------------------

/// Every interactive read/write the wizard performs. The real implementation
/// wraps `dialoguer`/`console`; the test implementation replays a scripted
/// answer queue. The state machine holds `&mut dyn Prompter` and never calls
/// stdin/stdout directly.
///
/// The semantic output methods (`heading`, `step`, `success`, …) all default to
/// `info`, so a test prompter only has to implement the four primitives and
/// still captures every line. `TtyPrompter` overrides them with color and a
/// small symbol vocabulary; the state machine chooses meaning by *which* method
/// it calls, never by formatting strings itself.
pub(crate) trait Prompter {
    /// Present `options` and return the chosen zero-based index.
    fn select(&mut self, prompt: &str, options: &[&str]) -> Result<usize>;
    /// Yes/no question with a default.
    fn confirm(&mut self, prompt: &str, default: bool) -> Result<bool>;
    /// Free-text input with a default shown to the user. Part of the mandated
    /// `Prompter` interface; reserved for interactive value entry (the settings
    /// step is a read-only inspector today, so it is not yet called).
    #[allow(dead_code)]
    fn input(&mut self, prompt: &str, default: &str) -> Result<String>;
    /// One line of body text.
    fn info(&mut self, message: &str);
    /// A blank spacer line.
    fn blank(&mut self) {
        self.info("");
    }
    /// A top-of-screen section heading.
    fn heading(&mut self, message: &str) {
        self.info(message);
    }
    /// A numbered get-started step header, e.g. `step("1", "Storage")`.
    fn step(&mut self, number: &str, title: &str) {
        self.info(&format!("{number} · {title}"));
    }
    /// A completed action.
    fn success(&mut self, message: &str) {
        self.info(&format!("✓ {message}"));
    }
    /// A skipped or secondary item.
    fn skip(&mut self, message: &str) {
        self.info(&format!("· {message}"));
    }
    /// A non-fatal attention line.
    fn warn(&mut self, message: &str) {
        self.info(&format!("! {message}"));
    }
    /// A failed step (the flow continues).
    fn failure(&mut self, message: &str) {
        self.info(&format!("✗ {message}"));
    }
    /// A dim, indented detail under a preceding line.
    fn note(&mut self, message: &str) {
        self.info(message);
    }
    /// An on/off state line, e.g. a configured integration.
    fn status(&mut self, on: bool, message: &str) {
        let mark = if on { "●" } else { "○" };
        self.info(&format!("{mark} {message}"));
    }
    /// A `label   value` row for summaries and the settings inspector.
    fn field(&mut self, label: &str, value: &str) {
        self.info(&format!("{label}  {value}"));
    }
    /// A copy-pasteable command, set off from prose.
    fn command(&mut self, command: &str) {
        self.info(&format!("  {command}"));
    }

    // --- Frame primitives. The state machine treats each screen as a redrawn
    // frame rather than appending to a scrollback log. These default to no-ops
    // (or `info`) so the scripted test prompter records the same content without
    // a terminal; only `TtyPrompter` actually clears and waits. ---

    /// Clear the screen so the next frame is drawn from the top. No-op off-TTY.
    fn clear(&mut self) {}
    /// A full-width divider between the chrome and the screen body.
    fn rule(&mut self) {}
    /// A standalone screen title for an action screen (e.g. `Health check`).
    /// Distinct from `step`, which is reserved for numbered get-started steps.
    fn screen_title(&mut self, title: &str) {
        self.heading(title);
    }
    /// The dim "press ↵" line that ends every action screen and waits for the
    /// user before the hub redraws. Off-TTY this returns immediately.
    fn pause(&mut self, _hint: &str) {}
}

/// Real `dialoguer`/`console` prompter for an attended terminal.
pub(crate) struct TtyPrompter {
    theme: dialoguer::theme::ColorfulTheme,
}

impl TtyPrompter {
    fn new() -> Self {
        use console::Style;
        use dialoguer::theme::ColorfulTheme;
        // A quiet, aligned theme: a 2-space gutter on every prompt, a cyan `?`,
        // and resolved prompts that recede into the scrollback instead of
        // echoing loudly.
        let theme = ColorfulTheme {
            prompt_prefix: console::style("  ?".to_string()).cyan().bold(),
            prompt_suffix: console::style("›".to_string()).dim(),
            success_prefix: console::style("  ·".to_string()).dim(),
            success_suffix: console::style("".to_string()),
            values_style: Style::new().dim(),
            active_item_prefix: console::style("  ❯".to_string()).cyan().bold(),
            inactive_item_prefix: console::style("   ".to_string()),
            active_item_style: Style::new().cyan().bold(),
            inactive_item_style: Style::new(),
            hint_style: Style::new().dim(),
            ..ColorfulTheme::default()
        };
        Self { theme }
    }
}

fn prompt_error(error: dialoguer::Error) -> Error {
    match error {
        dialoguer::Error::IO(source) => Error::Io {
            path: PathBuf::from("<wizard prompt>"),
            source,
        },
        #[allow(unreachable_patterns)]
        _ => Error::Validation("wizard prompt failed".to_string()),
    }
}

impl Prompter for TtyPrompter {
    fn select(&mut self, prompt: &str, options: &[&str]) -> Result<usize> {
        // `clear(true)` removes the menu after a pick and `report(false)`
        // suppresses the resolved-choice echo, so the menu leaves no scrollback
        // residue — the frame is redrawn cleanly on the next loop.
        dialoguer::Select::with_theme(&self.theme)
            .with_prompt(prompt)
            .items(options)
            .default(0)
            .clear(true)
            .report(false)
            .interact()
            .map_err(prompt_error)
    }

    fn confirm(&mut self, prompt: &str, default: bool) -> Result<bool> {
        // `report(false)`: don't leave a `· prompt  yes` echo stacking up under
        // each consent — the frame owns what stays on screen.
        dialoguer::Confirm::with_theme(&self.theme)
            .with_prompt(prompt)
            .default(default)
            .report(false)
            .interact()
            .map_err(prompt_error)
    }

    fn input(&mut self, prompt: &str, default: &str) -> Result<String> {
        dialoguer::Input::with_theme(&self.theme)
            .with_prompt(prompt)
            .default(default.to_string())
            .allow_empty(true)
            .interact_text()
            .map_err(prompt_error)
    }

    fn info(&mut self, message: &str) {
        if message.is_empty() {
            println!();
        } else {
            println!("  {message}");
        }
    }

    fn blank(&mut self) {
        println!();
    }

    fn heading(&mut self, message: &str) {
        println!("\n  {}", console::style(message).bold());
    }

    fn step(&mut self, number: &str, title: &str) {
        println!(
            "\n  {} {} {}",
            console::style(number).cyan().bold(),
            console::style("·").cyan().bold(),
            console::style(title).bold()
        );
    }

    fn success(&mut self, message: &str) {
        println!("    {} {message}", console::style("✓").green().bold());
    }

    fn skip(&mut self, message: &str) {
        println!(
            "    {} {}",
            console::style("·").dim(),
            console::style(message).dim()
        );
    }

    fn warn(&mut self, message: &str) {
        println!("    {} {message}", console::style("!").yellow().bold());
    }

    fn failure(&mut self, message: &str) {
        println!("    {} {message}", console::style("✗").red().bold());
    }

    fn note(&mut self, message: &str) {
        println!("      {}", console::style(message).dim());
    }

    fn status(&mut self, on: bool, message: &str) {
        let mark = if on {
            console::style("●").green()
        } else {
            console::style("○").dim()
        };
        println!("  {mark} {message}");
    }

    fn field(&mut self, label: &str, value: &str) {
        println!(
            "  {}  {value}",
            console::style(format_args!("{label:<13}")).dim()
        );
    }

    fn command(&mut self, command: &str) {
        println!("    {}", console::style(command).cyan());
    }

    fn clear(&mut self) {
        // Clears the visible region and homes the cursor; scrollback is
        // preserved, so each frame is clean but earlier frames remain scrollable.
        let _ = console::Term::stdout().clear_screen();
    }

    fn rule(&mut self) {
        let width = console::Term::stdout().size().1 as usize;
        // Inset by the 2-space gutter; cap so it never wraps on wide terminals.
        let len = width.saturating_sub(4).clamp(8, 72);
        println!("  {}", console::style("─".repeat(len)).dim());
    }

    fn screen_title(&mut self, title: &str) {
        println!("  {}\n", console::style(title).bold());
    }

    fn pause(&mut self, hint: &str) {
        println!("\n  {}", console::style(hint).dim());
        // Wait for a single keypress; ignore which key. Errors (e.g. EOF) just
        // fall through so the wizard never hangs on a closed stdin.
        let _ = console::Term::stdout().read_key();
    }
}

// ---------------------------------------------------------------------------
// Actions: every library call the wizard makes, injectable for spy tests.
// ---------------------------------------------------------------------------

/// Detection result for one upstream tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ToolState {
    pub tool: Tool,
    /// Tool found on `PATH` or already has nabu config present.
    pub present: bool,
    /// Capture hooks/plugin already installed for this tool.
    pub configured: bool,
    /// MCP server entry already present for this tool.
    pub mcp_configured: bool,
}

/// Read-only snapshot of effective settings for the inspector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SettingsView {
    pub home: PathBuf,
    pub opencode_server_url: Option<String>,
    pub semantic_feature_enabled: bool,
    pub semantic_available: bool,
    pub model_present: bool,
}

/// Every library function the wizard orchestrates. The live implementation
/// calls the same in-process functions the CLI subcommands call; tests inject a
/// spy that records calls and returns canned reports.
pub(crate) trait WizardActions {
    fn detect(&mut self, home: &Path) -> Result<Vec<ToolState>>;
    fn init_home(&mut self, home: &Path) -> Result<()>;
    fn install(&mut self, home: &Path, tool: Tool, dry_run: bool) -> Result<ConfigChangeReport>;
    fn uninstall(&mut self, home: &Path, tool: Tool, dry_run: bool) -> Result<ConfigChangeReport>;
    fn backfill_preview(&mut self, home: &Path) -> Result<BackfillDryRunReport>;
    fn backfill(&mut self, home: &Path) -> Result<BackfillReport>;
    fn doctor(&mut self, home: &Path) -> Result<DoctorReport>;
    fn mcp_install(&mut self, home: &Path, tool: Tool, dry_run: bool)
        -> Result<ConfigChangeReport>;
    fn sample_search(&mut self, home: &Path, query: &str) -> Result<usize>;
    fn settings(&mut self, home: &Path) -> Result<SettingsView>;
}

/// Live actions: the real in-process orchestration over existing functions.
pub(crate) struct LiveActions;

fn agent_tool(tool: Tool) -> AgentTool {
    match tool {
        Tool::Codex => AgentTool::Codex,
        Tool::Claude => AgentTool::Claude,
        Tool::Opencode => AgentTool::Opencode,
    }
}

impl WizardActions for LiveActions {
    fn detect(&mut self, home: &Path) -> Result<Vec<ToolState>> {
        let codex = codex_status(home)?;
        let claude = claude_status(home)?;
        let opencode = opencode_status(home)?;
        Ok(vec![
            ToolState {
                tool: Tool::Codex,
                present: codex.codex_installed || codex.hooks_installed,
                configured: codex.hooks_installed,
                mcp_configured: codex_mcp_entry_installed(),
            },
            ToolState {
                tool: Tool::Claude,
                present: claude.claude_installed || claude.hooks_installed,
                configured: claude.hooks_installed,
                mcp_configured: claude_mcp_entry_installed(),
            },
            ToolState {
                tool: Tool::Opencode,
                present: opencode.opencode_installed || opencode.plugin_installed,
                configured: opencode.plugin_installed,
                mcp_configured: opencode_mcp_entry_installed(),
            },
        ])
    }

    fn init_home(&mut self, home: &Path) -> Result<()> {
        init_home(home).map(|_| ())
    }

    fn install(&mut self, home: &Path, tool: Tool, dry_run: bool) -> Result<ConfigChangeReport> {
        match tool {
            Tool::Codex => install_codex(home, dry_run),
            Tool::Claude => install_claude(home, dry_run),
            Tool::Opencode => install_opencode(home, dry_run),
        }
    }

    fn uninstall(&mut self, home: &Path, tool: Tool, dry_run: bool) -> Result<ConfigChangeReport> {
        match tool {
            Tool::Codex => uninstall_codex(home, dry_run),
            Tool::Claude => uninstall_claude(home, dry_run),
            Tool::Opencode => uninstall_opencode(home, dry_run),
        }
    }

    fn backfill_preview(&mut self, home: &Path) -> Result<BackfillDryRunReport> {
        // Quiet emitter: the wizard renders its own progress through the
        // `Prompter` instead of letting the scan write telemetry to the screen.
        run_backfill_dry_run_command(
            home,
            BackfillTool::All,
            None,
            None,
            ProgressEmitter::quiet(),
        )
    }

    fn backfill(&mut self, home: &Path) -> Result<BackfillReport> {
        run_backfill_command(
            home,
            BackfillTool::All,
            None,
            None,
            ProgressEmitter::quiet(),
        )
    }

    fn doctor(&mut self, home: &Path) -> Result<DoctorReport> {
        Ok(doctor_with_options(home, false))
    }

    fn mcp_install(
        &mut self,
        home: &Path,
        tool: Tool,
        dry_run: bool,
    ) -> Result<ConfigChangeReport> {
        mcp_apply_one(home, agent_tool(tool), McpConfigAction::Install, dry_run)
    }

    fn sample_search(&mut self, home: &Path, query: &str) -> Result<usize> {
        let page = search_history_page(
            home,
            query,
            SearchOptions {
                limit: 3,
                ..SearchOptions::default()
            },
        )?;
        Ok(page.returned)
    }

    fn settings(&mut self, home: &Path) -> Result<SettingsView> {
        let server_url = opencode_server_url(home)?;
        let embed = embedding_model_status(home);
        Ok(SettingsView {
            home: home.to_path_buf(),
            opencode_server_url: server_url,
            semantic_feature_enabled: embed.feature_enabled,
            semantic_available: embed.semantic_available,
            model_present: embed.model_present,
        })
    }
}

// ---------------------------------------------------------------------------
// Entry point + TTY guard.
// ---------------------------------------------------------------------------

/// Run the wizard against a real terminal. Refuses (non-zero exit, mutating
/// nothing) when stdin/stdout is not a TTY.
pub(crate) fn run_wizard(home: &Path) -> Result<()> {
    ensure_interactive(
        std::io::stdin().is_terminal(),
        std::io::stdout().is_terminal(),
    )?;
    let mut prompter = TtyPrompter::new();
    let mut actions = LiveActions;
    run(&mut prompter, &mut actions, home)
}

/// TTY gate, split out so it is testable without a terminal. `Error::Validation`
/// maps to a non-zero exit code in `main`.
pub(crate) fn ensure_interactive(stdin_tty: bool, stdout_tty: bool) -> Result<()> {
    if stdin_tty && stdout_tty {
        Ok(())
    } else {
        Err(Error::Validation(NON_TTY_MESSAGE.to_string()))
    }
}

// ---------------------------------------------------------------------------
// State machine.
// ---------------------------------------------------------------------------

/// The wizard state machine. Pure orchestration over `prompter` + `actions`;
/// no direct IO, no direct config writes.
pub(crate) fn run(
    prompter: &mut dyn Prompter,
    actions: &mut dyn WizardActions,
    home: &Path,
) -> Result<()> {
    // The hub is a redrawn frame, not a growing log: every iteration clears and
    // repaints the chrome (brand + live status + rule) with the menu beneath it,
    // so the menu is always in the same place and nothing accumulates. Each
    // action runs on its own cleared screen and returns here on `↵`.
    loop {
        draw_chrome(prompter, actions, home)?;
        match prompter.select("What next?", &TOP_MENU)? {
            TOP_GET_STARTED => get_started(prompter, actions, home)?,
            TOP_MANAGE => manage_integrations(prompter, actions, home)?,
            TOP_BACKFILL => {
                action_screen(prompter, actions, home, "Backfill history", backfill_step)?
            }
            TOP_SETTINGS => action_screen(prompter, actions, home, "Settings", settings_body)?,
            TOP_HEALTH => action_screen(prompter, actions, home, "Health check", health_step)?,
            TOP_CONNECT => {
                draw_chrome(prompter, actions, home)?;
                prompter.screen_title("Connect agents (MCP)");
                let detected = actions.detect(home)?;
                mcp_register(prompter, actions, home, &detected)?;
                prompter.pause("↵ back");
            }
            TOP_QUIT => {
                quit_screen(prompter, actions, home)?;
                return Ok(());
            }
            _ => unreachable!("select returns an in-range index"),
        }
    }
}

/// Clear the screen and paint the constant chrome — brand, tagline, live status,
/// divider — at the top of every frame. Returns the detection snapshot so the
/// caller can reuse it without a second `detect()`.
fn draw_chrome(
    prompter: &mut dyn Prompter,
    actions: &mut dyn WizardActions,
    home: &Path,
) -> Result<Vec<ToolState>> {
    prompter.clear();
    prompter.heading("𒀭𒀝   nabu");
    prompter.info("Local, cross-agent history for Codex, Claude Code & OpenCode.");
    prompter.blank();

    let detected = actions.detect(home)?;
    let configured = joined_tool_labels(
        detected.iter().filter(|t| t.configured).map(|t| t.tool),
        " · ",
    );
    if configured.is_empty() {
        prompter.status(false, "Not set up yet.  Choose “Get started”.");
    } else {
        prompter.status(
            true,
            &format!("Capturing  {}      {}", configured, home.display()),
        );
    }
    prompter.blank();
    prompter.rule();
    prompter.blank();
    Ok(detected)
}

/// Run a read-only/management action on its own cleared frame: chrome, a screen
/// title, the action body, then a uniform `↵ back` so the result is read before
/// the hub redraws. Keeps action output from ever coexisting with the menu.
fn action_screen(
    prompter: &mut dyn Prompter,
    actions: &mut dyn WizardActions,
    home: &Path,
    title: &str,
    body: fn(&mut dyn Prompter, &mut dyn WizardActions, &Path) -> Result<()>,
) -> Result<()> {
    draw_chrome(prompter, actions, home)?;
    prompter.screen_title(title);
    body(prompter, actions, home)?;
    prompter.pause("↵ back");
    Ok(())
}

/// Final frame on quit: redraw chrome once and leave a plain sign-off in
/// scrollback (no pause — the wizard is exiting).
fn quit_screen(
    prompter: &mut dyn Prompter,
    actions: &mut dyn WizardActions,
    home: &Path,
) -> Result<()> {
    draw_chrome(prompter, actions, home)?;
    prompter.info("Done. Re-run `nabu wizard` any time.");
    Ok(())
}

/// The ordered first-run flow. Each step is skippable and mutates nothing
/// without an explicit confirm.
fn get_started(
    prompter: &mut dyn Prompter,
    actions: &mut dyn WizardActions,
    home: &Path,
) -> Result<()> {
    // One cleared frame for the whole linear flow: the numbered steps read
    // top-to-bottom (accumulation is correct here — it's a sequence, not a hub),
    // and a final `↵` returns to the redrawn menu.
    draw_chrome(prompter, actions, home)?;
    prompter.screen_title("Get started");
    prompter.info("Four steps. Each asks before it changes anything and backs up");
    prompter.info("any file it touches. Skip any step; re-run anytime.");

    // 1. Storage home.
    prompter.step("1", "Storage");
    if prompter.confirm(
        &format!("Create your history store at {}?", home.display()),
        true,
    )? {
        actions.init_home(home)?;
        prompter.success("Storage ready");
    } else {
        prompter.skip("Skipped — later steps need a store; re-run to create it.");
    }

    // 2. Detect tools, then per-tool capture.
    let detected = actions.detect(home)?;
    prompter.step("2", "Capture");
    if !detected.iter().any(|t| t.present) {
        prompter.skip("No Codex, Claude Code, or OpenCode install found — nothing to capture.");
    } else {
        prompter.info(&format!(
            "Found {}.",
            joined_tool_labels(detected.iter().filter(|t| t.present).map(|t| t.tool), ", ")
        ));
        prompter.info("Each install adds nabu capture hooks and backs the file up first.");
        prompter.info("See the exact change anytime:  nabu install <tool> --dry-run");
        for state in detected.iter().filter(|t| t.present) {
            install_step(prompter, actions, home, state)?;
        }
    }

    // 3. Backfill past history.
    prompter.step("3", "Backfill");
    backfill_step(prompter, actions, home)?;

    // Health check (read-only) — folded in, no own prompt.
    health_step(prompter, actions, home)?;

    // 4. MCP registration so agents can query history.
    prompter.step("4", "Connect");
    mcp_register(prompter, actions, home, &detected)?;

    // Sample search + "you're set" summary (re-detect for the true end state).
    summary(prompter, actions, home)?;
    prompter.pause("↵ back to menu");
    Ok(())
}

/// Preview + consent + install for a single tool. Already-configured tools are
/// reported and left untouched (no duplicate install).
fn install_step(
    prompter: &mut dyn Prompter,
    actions: &mut dyn WizardActions,
    home: &Path,
    state: &ToolState,
) -> Result<()> {
    let label = tool_label(state.tool);
    if state.configured {
        prompter.success(&format!("{label} capture already configured"));
        prompter.note("Manage integrations to repair or remove");
        return Ok(());
    }
    // Read-only preview to surface the target path; the full diff stays one
    // explicit command away rather than dumped inline.
    let preview = actions.install(home, state.tool, true)?;
    if prompter.confirm(
        &format!(
            "Install {label} capture?  → {}",
            preview.target_path.display()
        ),
        true,
    )? {
        match actions.install(home, state.tool, false) {
            Ok(_) => prompter.success(&format!("{label} capture installed")),
            Err(error) => {
                prompter.failure(&format!("{label} capture failed: {error}"));
                prompter.note("Other steps continue; fix and re-run to repair.");
            }
        }
    } else {
        prompter.skip(&format!("Skipped {label}"));
    }
    Ok(())
}

/// Coverage-diff preview, then optional real backfill on consent.
fn backfill_step(
    prompter: &mut dyn Prompter,
    actions: &mut dyn WizardActions,
    home: &Path,
) -> Result<()> {
    prompter.note("Scanning past sessions…");
    let preview = match actions.backfill_preview(home) {
        Ok(preview) => preview,
        Err(error) => {
            prompter.warn(&format!("Couldn’t scan past sessions: {error}"));
            return Ok(());
        }
    };
    if preview.source_files == 0 || preview.missing_events == 0 {
        prompter.skip("No past sessions to import — already up to date.");
        return Ok(());
    }
    if prompter.confirm(
        &format!(
            "Import {} from {} now?",
            plural(preview.missing_events, "event"),
            plural(preview.source_files, "past session"),
        ),
        true,
    )? {
        match actions.backfill(home) {
            Ok(report) => prompter.success(&format!(
                "Imported {} from {}",
                plural(report.appended_events, "event"),
                plural(report.source_files, "session"),
            )),
            Err(error) => prompter.failure(&format!("Backfill failed: {error}")),
        }
    } else {
        prompter.skip("Skipped backfill");
    }
    Ok(())
}

/// Run `doctor` (fast) and report health. Read-only — no consent needed.
fn health_step(
    prompter: &mut dyn Prompter,
    actions: &mut dyn WizardActions,
    home: &Path,
) -> Result<()> {
    match actions.doctor(home) {
        Ok(report) => {
            let line = format!(
                "Health: storage {} · index {} · backfill {}",
                ok_label(report.storage.ok),
                ok_label(report.index.ok),
                ok_label(report.backfill.ok),
            );
            if report.storage.ok && report.index.ok && report.backfill.ok {
                prompter.success(&line);
            } else {
                prompter.warn(&line);
                prompter.note("run  nabu doctor  for detail");
            }
        }
        Err(error) => prompter.failure(&format!("Health check failed: {error}")),
    }
    Ok(())
}

/// Offer MCP registration for the given tools with a single combined consent.
/// Tools already registered are skipped (no duplicate install).
fn mcp_register(
    prompter: &mut dyn Prompter,
    actions: &mut dyn WizardActions,
    home: &Path,
    tools: &[ToolState],
) -> Result<()> {
    let already = joined_tool_labels(
        tools
            .iter()
            .filter(|t| t.present && t.mcp_configured)
            .map(|t| t.tool),
        " · ",
    );
    if !already.is_empty() {
        prompter.success(&format!("Already connected: {already}"));
    }
    let pending = joined_tool_labels(
        tools
            .iter()
            .filter(|t| t.present && !t.mcp_configured)
            .map(|t| t.tool),
        ", ",
    );
    if pending.is_empty() {
        if already.is_empty() {
            prompter.skip("No detected agents to connect.");
        }
        return Ok(());
    }
    prompter.info("Register nabu as an MCP server so agents can search your history.");
    if prompter.confirm(&format!("Connect {pending}?"), true)? {
        let mut connected = String::new();
        for state in tools.iter().filter(|t| t.present && !t.mcp_configured) {
            match actions.mcp_install(home, state.tool, false) {
                Ok(_) => push_joined_tool_label(&mut connected, state.tool, " · "),
                Err(error) => prompter.failure(&format!(
                    "{} connect failed: {error}",
                    tool_label(state.tool)
                )),
            }
        }
        if !connected.is_empty() {
            prompter.success(&format!("Connected {connected}"));
        }
    } else {
        prompter.skip("Skipped — agents won’t search history until connected.");
    }
    Ok(())
}

/// One-screen "you're set" summary. Re-detects after the mutations above so it
/// reflects the *actual* end state, not the pre-install snapshot.
fn summary(
    prompter: &mut dyn Prompter,
    actions: &mut dyn WizardActions,
    home: &Path,
) -> Result<()> {
    let detected = actions.detect(home)?;
    let configured = joined_tool_labels(
        detected.iter().filter(|t| t.configured).map(|t| t.tool),
        " · ",
    );
    let connected = detected.iter().any(|t| t.mcp_configured);

    prompter.heading("✓ You’re set");
    if configured.is_empty() {
        prompter.info("No capture configured yet — install at least one integration above.");
    } else {
        prompter.field("Capturing", &configured);
        if connected {
            prompter.field("Searchable", "agents can query history over MCP");
        }
        prompter.field("Store", &home.display().to_string());
    }

    prompter.blank();
    prompter.info("Try it");
    prompter.command(SAMPLE_SEARCH_COMMAND);
    prompter.blank();
    prompter.info("Anytime");
    prompter.command("nabu wizard    this menu");
    prompter.command("nabu doctor    health check");

    if let Ok(view) = actions.settings(home) {
        if view.semantic_feature_enabled && !view.model_present {
            prompter.blank();
            prompter.info("Optional");
            prompter.note("Semantic search improves fuzzy recall, but needs a one-time model download and CPU-heavy embedding pass; later indexing embeds new history.");
            prompter.command("nabu embed download --yes");
        }
    }
    // Touch the sample-search action so the "you're set" screen reflects whether
    // history is already queryable; the count is informational only.
    let _ = actions.sample_search(home, SAMPLE_QUERY);
    Ok(())
}

/// Per-tool install/repair/uninstall menu. A sub-hub: each iteration redraws the
/// chrome and the aligned tool table; selecting a tool opens its own action frame.
fn manage_integrations(
    prompter: &mut dyn Prompter,
    actions: &mut dyn WizardActions,
    home: &Path,
) -> Result<()> {
    loop {
        let detected = draw_chrome(prompter, actions, home)?;
        prompter.screen_title("Manage integrations");
        let choice = match detected.as_slice() {
            [codex, claude, opencode] => {
                let labels = [
                    tool_status_label(codex),
                    tool_status_label(claude),
                    tool_status_label(opencode),
                ];
                let label_refs = [
                    labels[0].as_str(),
                    labels[1].as_str(),
                    labels[2].as_str(),
                    "Back",
                ];
                prompter.select("Select a tool", &label_refs)?
            }
            states => {
                let mut labels: Vec<String> = states.iter().map(tool_status_label).collect();
                labels.push("Back".to_string());
                let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
                prompter.select("Select a tool", &label_refs)?
            }
        };
        if choice == detected.len() {
            return Ok(());
        }
        let state = detected[choice];

        // The chosen tool gets its own frame: chrome, the tool name as the title,
        // then repair/remove. Result is read on `↵` before the table redraws.
        draw_chrome(prompter, actions, home)?;
        prompter.screen_title(tool_label(state.tool));
        let actions_menu = ["Repair / reinstall capture", "Remove capture", "Back"];
        match prompter.select("What next?", &actions_menu)? {
            0 => {
                // Force a preview+consent even when already configured (repair).
                let repair_state = ToolState {
                    configured: false,
                    ..state
                };
                install_step(prompter, actions, home, &repair_state)?;
                prompter.pause("↵ back");
            }
            1 => {
                uninstall_step(prompter, actions, home, &state)?;
                prompter.pause("↵ back");
            }
            _ => {}
        }
    }
}

/// Preview + consent + uninstall for a single tool.
fn uninstall_step(
    prompter: &mut dyn Prompter,
    actions: &mut dyn WizardActions,
    home: &Path,
    state: &ToolState,
) -> Result<()> {
    let label = tool_label(state.tool);
    let preview = actions.uninstall(home, state.tool, true)?;
    prompter.info(&format!(
        "Removes only nabu entries from {} (backed up first).",
        preview.target_path.display()
    ));
    if prompter.confirm(&format!("Remove nabu capture for {label}?"), false)? {
        match actions.uninstall(home, state.tool, false) {
            Ok(_) => prompter.success(&format!("{label} capture removed")),
            Err(error) => prompter.failure(&format!("{label} removal failed: {error}")),
        }
    } else {
        prompter.skip(&format!("Kept {label}"));
    }
    Ok(())
}

/// Read-only settings inspector: reports effective configuration and exactly
/// how to change each value. Writes nothing — it never flips redaction and adds
/// no new config write path. The screen title/chrome are supplied by
/// `action_screen`; this renders only the body.
fn settings_body(
    prompter: &mut dyn Prompter,
    actions: &mut dyn WizardActions,
    home: &Path,
) -> Result<()> {
    prompter.note("read-only — shows where to change each value");
    let view = actions.settings(home)?;

    prompter.field("Storage home", &view.home.display().to_string());
    prompter.note("--home <path>  or  NABU_HOME");

    match &view.opencode_server_url {
        Some(url) => prompter.field("OpenCode sync", &format!("on ({url})")),
        None => prompter.field("OpenCode sync", "off"),
    }
    prompter.note(&format!(
        "set [opencode] server_url in {}",
        home.join("config.toml").display()
    ));

    prompter.field("Redaction", "opt-in — never on by default");
    prompter.note("--redact on export, or redact=true via MCP");

    match (view.semantic_feature_enabled, view.model_present) {
        (false, _) => prompter.field(
            "Semantic search",
            "not built (rebuild with --features semantic)",
        ),
        (true, false) => prompter.field("Semantic search", "built · model not downloaded"),
        (true, true) => {
            let semantic = format!("ready (available={})", view.semantic_available);
            prompter.field("Semantic search", &semantic);
        }
    }
    if view.semantic_feature_enabled && !view.model_present {
        prompter.note("better fuzzy recall; one-time model download plus CPU-heavy embedding");
        prompter.note("nabu embed download --yes");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Small helpers.
// ---------------------------------------------------------------------------

fn ok_label(ok: bool) -> &'static str {
    if ok {
        "ok"
    } else {
        "needs attention"
    }
}

/// `1 event` / `2 events` — count with a naive pluralized noun.
fn plural(n: usize, singular: &str) -> Plural<'_> {
    Plural { n, singular }
}

struct Plural<'a> {
    n: usize,
    singular: &'a str,
}

impl std::fmt::Display for Plural<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.n == 1 {
            write!(formatter, "1 {}", self.singular)
        } else {
            write!(formatter, "{} {}s", self.n, self.singular)
        }
    }
}

/// Human display name for a tool — the wizard speaks these, not the lowercase
/// internal identifiers.
fn tool_label(tool: Tool) -> &'static str {
    match tool {
        Tool::Codex => "Codex",
        Tool::Claude => "Claude Code",
        Tool::Opencode => "OpenCode",
    }
}

fn joined_tool_labels(tools: impl IntoIterator<Item = Tool>, separator: &str) -> String {
    let mut labels = String::new();
    for tool in tools {
        push_joined_tool_label(&mut labels, tool, separator);
    }
    labels
}

fn push_joined_tool_label(labels: &mut String, tool: Tool, separator: &str) {
    if !labels.is_empty() {
        labels.push_str(separator);
    }
    labels.push_str(tool_label(tool));
}

/// One scannable row per tool for the Manage menu: name, then capture/mcp/PATH
/// state as filled/empty dots.
fn tool_status_label(state: &ToolState) -> String {
    let dot = |on: bool| if on { "●" } else { "○" };
    let presence = if state.present {
        "detected"
    } else {
        "not found"
    };
    format!(
        "{:<13} {} capture   {} mcp     {presence}",
        tool_label(state.tool),
        dot(state.configured),
        dot(state.mcp_configured),
    )
}

// Re-exported so the crate-level integration test (which reuses the shared
// `ENV_LOCK`) can drive the live actions with a scripted prompter.
#[cfg(test)]
pub(crate) use tests::ScriptedPrompter;

#[cfg(test)]
mod tests {
    use super::*;
    use nabu_core::{CoverageSummary, DoctorCheck, StorageFootprint};
    use std::collections::VecDeque;

    /// Test prompter: replays scripted answers and records info output.
    pub(crate) struct ScriptedPrompter {
        selects: VecDeque<usize>,
        confirms: VecDeque<bool>,
        #[allow(dead_code)]
        inputs: VecDeque<String>,
        pub info_log: Vec<String>,
        /// How many times the wizard cleared the screen to start a fresh frame.
        pub clears: usize,
    }

    impl ScriptedPrompter {
        pub(crate) fn new() -> Self {
            Self {
                selects: VecDeque::new(),
                confirms: VecDeque::new(),
                inputs: VecDeque::new(),
                info_log: Vec::new(),
                clears: 0,
            }
        }

        pub(crate) fn selects(mut self, values: impl IntoIterator<Item = usize>) -> Self {
            self.selects = values.into_iter().collect();
            self
        }

        pub(crate) fn confirms(mut self, values: impl IntoIterator<Item = bool>) -> Self {
            self.confirms = values.into_iter().collect();
            self
        }
    }

    impl Prompter for ScriptedPrompter {
        fn select(&mut self, _prompt: &str, options: &[&str]) -> Result<usize> {
            let index = self
                .selects
                .pop_front()
                .expect("scripted prompter ran out of select answers");
            assert!(index < options.len(), "scripted select index out of range");
            Ok(index)
        }

        fn confirm(&mut self, _prompt: &str, _default: bool) -> Result<bool> {
            Ok(self
                .confirms
                .pop_front()
                .expect("scripted prompter ran out of confirm answers"))
        }

        fn input(&mut self, _prompt: &str, default: &str) -> Result<String> {
            Ok(self
                .inputs
                .pop_front()
                .unwrap_or_else(|| default.to_string()))
        }

        fn info(&mut self, message: &str) {
            self.info_log.push(message.to_string());
        }

        fn clear(&mut self) {
            self.clears += 1;
        }
    }

    /// Spy actions: records every call and returns canned reports.
    pub(crate) struct SpyActions {
        pub calls: Vec<String>,
        detected: Vec<ToolState>,
    }

    impl SpyActions {
        fn new(detected: Vec<ToolState>) -> Self {
            Self {
                calls: Vec::new(),
                detected,
            }
        }

        fn all_present_unconfigured() -> Self {
            Self::new(
                Tool::all()
                    .into_iter()
                    .map(|tool| ToolState {
                        tool,
                        present: true,
                        configured: false,
                        mcp_configured: false,
                    })
                    .collect(),
            )
        }

        fn all_present_configured() -> Self {
            Self::new(
                Tool::all()
                    .into_iter()
                    .map(|tool| ToolState {
                        tool,
                        present: true,
                        configured: true,
                        mcp_configured: true,
                    })
                    .collect(),
            )
        }

        /// Only the mutating, non-dry-run calls, normalized for assertions.
        fn mutating_calls(&self) -> Vec<String> {
            self.calls
                .iter()
                .filter_map(|call| match call.as_str() {
                    "init_home" | "backfill" => Some(call.clone()),
                    other if other.ends_with(":dry=false") => {
                        Some(other.trim_end_matches(":dry=false").to_string())
                    }
                    _ => None,
                })
                .collect()
        }
    }

    fn canned_report(tool: Tool, dry_run: bool) -> ConfigChangeReport {
        ConfigChangeReport {
            tool,
            target_path: PathBuf::from(format!("/canned/{tool}")),
            changed: !dry_run,
            dry_run,
            summary: format!("canned {tool} report"),
            diff: "--- before\n--- after\n".to_string(),
        }
    }

    fn canned_doctor() -> DoctorReport {
        DoctorReport {
            level: "ok".to_string(),
            integrity: "ok".to_string(),
            storage: DoctorCheck {
                ok: true,
                message: "ok".to_string(),
            },
            index: DoctorCheck {
                ok: true,
                message: "ok".to_string(),
            },
            backfill: DoctorCheck {
                ok: true,
                message: "ok".to_string(),
            },
            coverage: CoverageSummary {
                checkpointed_sources: 0,
                captured_sessions: 0,
                captured_events: 0,
            },
            storage_footprint: StorageFootprint {
                raw_bytes: 0,
                index_bytes: 0,
                vectors_bytes: 0,
                spool_bytes: 0,
                blobs_bytes: 0,
                models_bytes: 0,
                canonical_total: 0,
                derived_total: 0,
                total_bytes: 0,
            },
            latest_captured_events: Default::default(),
            stats: None,
        }
    }

    impl WizardActions for SpyActions {
        fn detect(&mut self, _home: &Path) -> Result<Vec<ToolState>> {
            self.calls.push("detect".to_string());
            Ok(self.detected.clone())
        }

        fn init_home(&mut self, _home: &Path) -> Result<()> {
            self.calls.push("init_home".to_string());
            Ok(())
        }

        fn install(
            &mut self,
            _home: &Path,
            tool: Tool,
            dry_run: bool,
        ) -> Result<ConfigChangeReport> {
            self.calls.push(format!("install:{tool}:dry={dry_run}"));
            Ok(canned_report(tool, dry_run))
        }

        fn uninstall(
            &mut self,
            _home: &Path,
            tool: Tool,
            dry_run: bool,
        ) -> Result<ConfigChangeReport> {
            self.calls.push(format!("uninstall:{tool}:dry={dry_run}"));
            Ok(canned_report(tool, dry_run))
        }

        fn backfill_preview(&mut self, _home: &Path) -> Result<BackfillDryRunReport> {
            self.calls.push("backfill_preview".to_string());
            Ok(BackfillDryRunReport {
                source_files: 2,
                on_disk_events: 10,
                captured_events: 5,
                missing_events: 5,
                partial_sessions: 1,
                sessions: Vec::new(),
            })
        }

        fn backfill(&mut self, _home: &Path) -> Result<BackfillReport> {
            self.calls.push("backfill".to_string());
            Ok(BackfillReport {
                source_files: 2,
                appended_events: 5,
                checkpoint_files: 2,
                discontinuities: 0,
            })
        }

        fn doctor(&mut self, _home: &Path) -> Result<DoctorReport> {
            self.calls.push("doctor".to_string());
            Ok(canned_doctor())
        }

        fn mcp_install(
            &mut self,
            _home: &Path,
            tool: Tool,
            dry_run: bool,
        ) -> Result<ConfigChangeReport> {
            self.calls.push(format!("mcp_install:{tool}:dry={dry_run}"));
            Ok(canned_report(tool, dry_run))
        }

        fn sample_search(&mut self, _home: &Path, _query: &str) -> Result<usize> {
            self.calls.push("sample_search".to_string());
            Ok(0)
        }

        fn settings(&mut self, _home: &Path) -> Result<SettingsView> {
            self.calls.push("settings".to_string());
            Ok(SettingsView {
                home: PathBuf::from("/canned/home"),
                opencode_server_url: None,
                semantic_feature_enabled: false,
                semantic_available: false,
                model_present: false,
            })
        }
    }

    const HOME: &str = "/canned/home";

    #[test]
    fn tty_guard_refuses_non_terminal_and_allows_terminal() {
        assert!(ensure_interactive(true, true).is_ok());
        assert!(ensure_interactive(false, true).is_err());
        assert!(ensure_interactive(true, false).is_err());
        assert!(ensure_interactive(false, false).is_err());
    }

    #[test]
    fn full_consent_get_started_matches_init_install_all_backfill_mcp() {
        // Get started, then Quit.
        let mut prompter = ScriptedPrompter::new()
            .selects([TOP_GET_STARTED, TOP_QUIT])
            // init, 3 installs, backfill, mcp register.
            .confirms([true, true, true, true, true, true]);
        let mut actions = SpyActions::all_present_unconfigured();

        run(&mut prompter, &mut actions, Path::new(HOME)).unwrap();

        assert_eq!(
            actions.mutating_calls(),
            vec![
                "init_home",
                "install:codex",
                "install:claude",
                "install:opencode",
                "backfill",
                "mcp_install:codex",
                "mcp_install:claude",
                "mcp_install:opencode",
            ]
        );
    }

    #[test]
    fn declining_every_step_changes_nothing() {
        let mut prompter = ScriptedPrompter::new()
            .selects([TOP_GET_STARTED, TOP_QUIT])
            .confirms([false, false, false, false, false, false]);
        let mut actions = SpyActions::all_present_unconfigured();

        run(&mut prompter, &mut actions, Path::new(HOME)).unwrap();

        assert!(
            actions.mutating_calls().is_empty(),
            "no mutating calls expected, got {:?}",
            actions.mutating_calls()
        );
        // Read-only previews still ran.
        assert!(actions.calls.iter().any(|c| c == "install:codex:dry=true"));
        assert!(actions.calls.iter().any(|c| c == "backfill_preview"));
        assert!(actions.calls.iter().any(|c| c == "doctor"));
    }

    #[test]
    fn rerun_on_configured_home_does_not_duplicate_installs() {
        let mut prompter = ScriptedPrompter::new()
            .selects([TOP_GET_STARTED, TOP_QUIT])
            // init + backfill are the only confirms reached when all tools are
            // already configured and MCP already registered.
            .confirms([true, true]);
        let mut actions = SpyActions::all_present_configured();

        run(&mut prompter, &mut actions, Path::new(HOME)).unwrap();

        let mutating = actions.mutating_calls();
        assert!(
            !mutating.iter().any(|c| c.starts_with("install:")),
            "configured tools must not be reinstalled, got {mutating:?}"
        );
        assert!(
            !mutating.iter().any(|c| c.starts_with("mcp_install:")),
            "already-registered MCP must not be re-registered, got {mutating:?}"
        );
        assert_eq!(mutating, vec!["init_home", "backfill"]);
    }

    #[test]
    fn health_menu_entry_runs_doctor() {
        let mut prompter = ScriptedPrompter::new().selects([TOP_HEALTH, TOP_QUIT]);
        let mut actions = SpyActions::all_present_configured();
        run(&mut prompter, &mut actions, Path::new(HOME)).unwrap();
        assert!(actions.calls.iter().any(|c| c == "doctor"));
    }

    #[test]
    fn each_screen_redraws_a_fresh_frame() {
        // The hub is a redrawn frame, not an append-only log: opening one action
        // (Health check) and returning must clear at least three times — the hub
        // before the action, the action screen, and the hub on re-entry before
        // quit. This is the regression guard for the screenshot defect where
        // menus and results accumulated on one screen.
        let mut prompter = ScriptedPrompter::new().selects([TOP_HEALTH, TOP_QUIT]);
        let mut actions = SpyActions::all_present_configured();
        run(&mut prompter, &mut actions, Path::new(HOME)).unwrap();
        assert!(
            prompter.clears >= 3,
            "expected the hub and action to clear into fresh frames, got {}",
            prompter.clears
        );
    }

    #[test]
    fn backfill_menu_entry_runs_backfill_on_consent() {
        let mut prompter = ScriptedPrompter::new()
            .selects([TOP_BACKFILL, TOP_QUIT])
            .confirms([true]);
        let mut actions = SpyActions::all_present_configured();
        run(&mut prompter, &mut actions, Path::new(HOME)).unwrap();
        assert!(actions.calls.iter().any(|c| c == "backfill_preview"));
        assert!(actions.calls.iter().any(|c| c == "backfill"));
    }

    #[test]
    fn connect_menu_entry_registers_mcp_on_consent() {
        let mut prompter = ScriptedPrompter::new()
            .selects([TOP_CONNECT, TOP_QUIT])
            .confirms([true]);
        let mut actions = SpyActions::all_present_unconfigured();
        run(&mut prompter, &mut actions, Path::new(HOME)).unwrap();
        assert_eq!(
            actions
                .calls
                .iter()
                .filter(|c| c.ends_with(":dry=false") && c.starts_with("mcp_install:"))
                .count(),
            3
        );
    }

    #[test]
    fn settings_menu_entry_is_read_only() {
        let mut prompter = ScriptedPrompter::new().selects([TOP_SETTINGS, TOP_QUIT]);
        let mut actions = SpyActions::all_present_configured();
        run(&mut prompter, &mut actions, Path::new(HOME)).unwrap();
        assert!(actions.calls.iter().any(|c| c == "settings"));
        assert!(
            actions.mutating_calls().is_empty(),
            "settings inspector must not mutate, got {:?}",
            actions.mutating_calls()
        );
    }

    #[test]
    fn manage_install_routes_through_install_function() {
        let mut prompter = ScriptedPrompter::new()
            // Manage; select codex (0); Install/repair (0); Back (3); Quit.
            .selects([TOP_MANAGE, 0, 0, 3, TOP_QUIT])
            .confirms([true]);
        let mut actions = SpyActions::all_present_unconfigured();
        run(&mut prompter, &mut actions, Path::new(HOME)).unwrap();
        assert!(actions.calls.iter().any(|c| c == "install:codex:dry=false"));
    }

    #[test]
    fn manage_uninstall_routes_through_uninstall_function() {
        let mut prompter = ScriptedPrompter::new()
            // Manage; select claude (1); Uninstall (1); Back (3); Quit.
            .selects([TOP_MANAGE, 1, 1, 3, TOP_QUIT])
            .confirms([true]);
        let mut actions = SpyActions::all_present_configured();
        run(&mut prompter, &mut actions, Path::new(HOME)).unwrap();
        assert!(actions
            .calls
            .iter()
            .any(|c| c == "uninstall:claude:dry=false"));
    }
}
