use nabu_core::{
    doctor_with_options, export_session_jsonl_with_options, export_session_markdown_with_options,
    get_event_by_pointer_with_options, get_session_page, list_sessions, redact_export_json,
    redact_export_text, search_history_page, Error, EventOptions, SearchMode, SearchOptions,
    SearchResult, SessionOptions, StoredEvent, Tool,
};
use serde_json::{json, Value};
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::mpsc::{channel, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread;

pub use nabu_core as core;

const PROTOCOL_VERSION: &str = "2025-03-26";
const MAX_MCP_BYTES: usize = 256 * 1024;
const JSON_FIT_SAFETY_BYTES: usize = 2048;
const DEFAULT_MCP_DEEP_DOCTOR_MAX_BYTES: u64 = 500 * 1024 * 1024;

/// Fallback ceiling on MCP requests handled in parallel when the host CPU count
/// is unavailable. Coding agents routinely issue several tool calls at once.
const DEFAULT_MAX_CONCURRENCY: usize = 8;

/// How many MCP requests may be handled concurrently. `NABU_MCP_MAX_CONCURRENCY`
/// overrides; otherwise scale to the host CPU count, clamped to a sane band.
fn max_concurrency() -> usize {
    if let Some(value) = std::env::var("NABU_MCP_MAX_CONCURRENCY")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|&n| n > 0)
    {
        return value;
    }
    thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(DEFAULT_MAX_CONCURRENCY)
        .clamp(2, 16)
}

fn mcp_deep_doctor_max_bytes() -> Option<u64> {
    match std::env::var("NABU_MCP_DEEP_DOCTOR_MAX_BYTES")
        .ok()
        .and_then(|raw| raw.parse::<u64>().ok())
    {
        Some(0) => None,
        Some(value) => Some(value),
        None => Some(DEFAULT_MCP_DEEP_DOCTOR_MAX_BYTES),
    }
}

#[derive(Debug, Clone)]
struct ResponseBudget {
    used_bytes: usize,
    limit_bytes: usize,
}

impl ResponseBudget {
    fn from_base(base: &Value) -> Option<Self> {
        let used_bytes = serde_json::to_vec(base).ok()?.len();
        Some(Self {
            used_bytes,
            limit_bytes: MAX_MCP_BYTES.saturating_sub(JSON_FIT_SAFETY_BYTES),
        })
    }

    fn try_reserve_value(&mut self, value: &Value, separator_bytes: usize) -> bool {
        let Ok(serialized) = serde_json::to_vec(value) else {
            return false;
        };
        let next = self
            .used_bytes
            .saturating_add(separator_bytes)
            .saturating_add(serialized.len());
        if next > self.limit_bytes {
            return false;
        }
        self.used_bytes = next;
        true
    }
}

#[derive(Debug, Clone)]
struct RecallWindow {
    start: i64,
    end: i64,
    events: Option<Vec<StoredEvent>>,
}

pub fn serve_stdio(home: PathBuf) -> nabu_core::Result<()> {
    // Pass the unlocked `Stdout` (it is `Send`, unlike `StdoutLock`) so the
    // writer can run on its own thread; the reader keeps the `StdinLock` on this
    // thread, where `Send` is not required.
    serve_with_io(home, std::io::stdin().lock(), std::io::stdout())
}

pub fn serve_with_io<R, W>(home: PathBuf, reader: R, writer: W) -> nabu_core::Result<()>
where
    R: BufRead,
    W: Write + Send,
{
    // Concurrent request handling. A serial loop made every later request wait
    // behind the current one, so one slow tool call (e.g. a multi-minute
    // `history_doctor` deep integrity scan on a multi-GB index, or a large
    // export) stalled all search/list calls and clients timed out. JSON-RPC
    // permits out-of-order responses keyed by `id`, handlers are stateless, and
    // each opens its own SQLite connection against a WAL index, so requests are
    // safe to run in parallel.
    //
    // Layout: this thread reads stdin and feeds a bounded work queue; a pool of
    // worker threads handle requests; a single writer thread serializes every
    // response onto the output stream (one consumer means responses never
    // interleave). `thread::scope` lets the writer borrow non-`'static` writers.
    let workers = max_concurrency();
    // Bound the queue so a flood of input cannot buffer without limit: once the
    // pool is saturated the reader blocks here, applying backpressure upstream.
    let (work_tx, work_rx) = sync_channel::<String>(workers);
    let work_rx = Arc::new(Mutex::new(work_rx));
    let (resp_tx, resp_rx) = channel::<Value>();
    let home = &home;

    thread::scope(|scope| -> nabu_core::Result<()> {
        let writer_handle = scope.spawn(move || -> nabu_core::Result<()> {
            let mut writer = writer;
            for response in resp_rx {
                serde_json::to_writer(&mut writer, &response)?;
                writer.write_all(b"\n").map_err(|source| Error::Io {
                    path: PathBuf::from("<stdout>"),
                    source,
                })?;
                writer.flush().map_err(|source| Error::Io {
                    path: PathBuf::from("<stdout>"),
                    source,
                })?;
            }
            Ok(())
        });

        for _ in 0..workers {
            let work_rx = Arc::clone(&work_rx);
            let resp_tx = resp_tx.clone();
            scope.spawn(move || loop {
                // Hold the queue lock only to dequeue; a closed channel (reader
                // done) returns `Err`, which ends the worker.
                let line = {
                    let rx = work_rx.lock().expect("work queue mutex poisoned");
                    rx.recv()
                };
                let Ok(line) = line else { break };
                match handle_message(home, &line) {
                    Ok(Some(response)) => {
                        // A send error means the writer thread is gone (shutdown).
                        if resp_tx.send(response).is_err() {
                            break;
                        }
                    }
                    Ok(None) => {}
                    Err(error) => eprintln!("nabu mcp error: {error}"),
                }
            });
        }
        // Drop this thread's response sender so the writer can finish once every
        // worker has dropped its clone.
        drop(resp_tx);

        let mut read_result = Ok(());
        for line in reader.lines() {
            let line = match line {
                Ok(line) => line,
                Err(source) => {
                    read_result = Err(Error::Io {
                        path: PathBuf::from("<stdin>"),
                        source,
                    });
                    break;
                }
            };
            if line.trim().is_empty() {
                continue;
            }
            // A send error means every worker has exited; stop reading.
            if work_tx.send(line).is_err() {
                break;
            }
        }
        // Closing the work queue drains the pool; workers then drop their
        // response senders, which lets the writer thread reach EOF and return.
        drop(work_tx);

        let write_result = writer_handle.join().expect("writer thread panicked");
        read_result.and(write_result)
    })
}

fn handle_message(home: &Path, line: &str) -> nabu_core::Result<Option<Value>> {
    let request: Value = serde_json::from_str(line)?;
    let id = request.get("id").cloned();
    let method = request
        .get("method")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::Validation("json-rpc method is required".to_string()))?;

    if id.is_none() {
        return Ok(None);
    }

    let id = id.expect("checked");
    let params = request.get("params").cloned().unwrap_or_else(|| json!({}));
    let response = match method {
        "initialize" => ok_response(
            id,
            json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {
                    "tools": {},
                    "resources": {},
                    "prompts": {}
                },
                "serverInfo": {
                    "name": "nabu",
                    "version": env!("CARGO_PKG_VERSION")
                }
            }),
        ),
        "ping" => ok_response(id, json!({})),
        "tools/list" => ok_response(id, json!({ "tools": tool_descriptions() })),
        "tools/call" => ok_response(id, handle_tool_call(home, &params)),
        "resources/list" => ok_response(id, json!({ "resources": resource_descriptions() })),
        "resources/read" => ok_response(id, handle_resource_read(home, &params)),
        "prompts/list" => ok_response(id, json!({ "prompts": prompt_descriptions() })),
        "prompts/get" => ok_response(id, handle_prompt_get(&params)),
        _ => error_response(
            id,
            -32601,
            "method not found",
            json!({
                "code": "method_not_found",
                "message": format!("unsupported MCP method: {method}"),
                "recoverable": true
            }),
        ),
    };
    Ok(Some(response))
}

fn ok_response(id: Value, result: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result
    })
}

fn error_response(id: Value, code: i64, message: &str, data: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {
            "code": code,
            "message": message,
            "data": data
        }
    })
}

fn tool_descriptions() -> Value {
    json!([
        {
            "name": "search_history",
            "description": "Search indexed local agent history citation-first: returns score, snippet, tool, session_id, raw_line, and payload=null by default. Drill into hits with get_session around_raw_line/before/after, get_event, or include_payload=true for full payloads. Page with offset; include_deltas restores deltas; dedupe=false restores adjacent twin rows. Set corroborate=true to add local read-only git existence checks for mentioned commits, branches, and files; PR refs are reported unresolved/needs_network and never fetched.",
            "inputSchema": tool_schema("search_history")
        },
        {
            "name": "list_sessions",
            "description": "List recent captured sessions with counts and raw-file pointers.",
            "inputSchema": tool_schema("list_sessions")
        },
        {
            "name": "get_session",
            "description": "Read a faithful non-deduped page or context window from one session. Use around_raw_line with before/after to inspect context around a search hit; include_deltas=true restores assistant deltas. Set corroborate=true for local read-only git annotations; PR refs require network and remain unresolved.",
            "inputSchema": tool_schema("get_session")
        },
        {
            "name": "export_session",
            "description": "Export one session as Markdown or JSONL for agent handoff.",
            "inputSchema": tool_schema("export_session")
        },
        {
            "name": "get_event",
            "description": "Read one raw envelope and normalized text by raw pointer. Set corroborate=true for local read-only git annotations; PR refs require network and remain unresolved.",
            "inputSchema": tool_schema("get_event")
        },
        {
            "name": "history_doctor",
            "description": "Report fast local health by default using an O(1) structural index check and latest-event citations; pass deep=true for full SQLite integrity (scans the whole index) and counts. Over MCP, deep=true is refused when the index exceeds NABU_MCP_DEEP_DOCTOR_MAX_BYTES (default 500 MiB).",
            "inputSchema": tool_schema("history_doctor")
        },
        {
            "name": "recall_answer",
            "description": "Assemble cited context windows for a query in one read-only call. It does not write an answer, call an LLM, mutate history, or make network requests. Set corroborate=true to annotate hits/context with local read-only git existence checks; PR refs remain unresolved/needs_network.",
            "inputSchema": tool_schema("recall_answer")
        }
    ])
}

fn handle_tool_call(home: &Path, params: &Value) -> Value {
    let name = params.get("name").and_then(Value::as_str).unwrap_or("");
    let arguments = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));

    match call_tool(home, name, &arguments) {
        Ok(structured) => tool_success(name, structured),
        Err(error) => tool_failure(error),
    }
}

fn call_tool(home: &Path, name: &str, arguments: &Value) -> Result<Value, ToolError> {
    match name {
        "search_history" => tool_search_history(home, arguments),
        "list_sessions" => tool_list_sessions(home, arguments),
        "get_session" => tool_get_session(home, arguments),
        "export_session" => tool_export_session(home, arguments),
        "get_event" => tool_get_event(home, arguments),
        "history_doctor" => tool_history_doctor(home, arguments),
        "recall_answer" => tool_recall_answer(home, arguments),
        _ => Err(ToolError::new(
            "VALIDATION_ERROR",
            format!("unknown MCP tool: {name}"),
            true,
        )),
    }
}

fn tool_success(name: &str, structured: Value) -> Value {
    let summary = concise_summary(name, &structured);
    let allow_oversized = name == "export_session"
        && structured.get("format").and_then(Value::as_str) == Some("jsonl");
    let structured = enforce_size_bound(structured, allow_oversized);
    json!({
        "content": [
            {
                "type": "text",
                "text": summary
            }
        ],
        "structuredContent": structured,
        "isError": false
    })
}

fn tool_failure(error: ToolError) -> Value {
    let error = json!({
        "code": error.code,
        "message": error.message,
        "recoverable": error.recoverable,
        "hint": error.hint,
        "details": error.details
    });
    json!({
        "content": [
            {
                "type": "text",
                "text": error["message"]
            }
        ],
        "structuredContent": {
            "ok": false,
            "error": error
        },
        "isError": true
    })
}

fn tool_search_history(home: &Path, arguments: &Value) -> Result<Value, ToolError> {
    let query = required_string(arguments, "query")?;
    if query.trim().is_empty() {
        return Err(ToolError::new(
            "VALIDATION_ERROR",
            "query must not be empty",
            true,
        ));
    }
    let limit = clamped_usize(arguments, "limit", 10, 1, 50)?;
    let offset = optional_usize_min(arguments, "offset", 0)?.unwrap_or(0);
    let max_snippet_chars = clamped_usize(arguments, "max_snippet_chars", 240, 1, 1000)?;
    let options = SearchOptions {
        tool: optional_tool(arguments, "tool")?,
        session_id: optional_string(arguments, "session_id"),
        cwd: optional_string(arguments, "cwd"),
        since: optional_string(arguments, "since"),
        canonical_type: optional_string(arguments, "canonical_type"),
        file: optional_string(arguments, "file"),
        command: optional_string(arguments, "command"),
        limit,
        offset,
        include_payload: optional_bool(arguments, "include_payload", false),
        include_deltas: optional_bool(arguments, "include_deltas", false),
        dedupe: optional_bool(arguments, "dedupe", true),
        max_snippet_chars,
        mode: optional_search_mode(arguments)?,
        corroborate: optional_bool(arguments, "corroborate", false),
    };
    let redact = optional_bool(arguments, "redact", false);
    let mut value = serde_json::to_value(search_history_page(home, query, options)?)?;
    if redact {
        redact_result_snippets(&mut value);
    }
    Ok(fit_search_response(value))
}

fn tool_list_sessions(home: &Path, arguments: &Value) -> Result<Value, ToolError> {
    let limit = bounded_usize(arguments, "limit", 20, 1, 100)?;
    let sessions = list_sessions(
        home,
        optional_tool(arguments, "tool")?,
        optional_string(arguments, "cwd").as_deref(),
        optional_string(arguments, "since").as_deref(),
        limit,
    )?;
    Ok(json!({ "sessions": sessions }))
}

fn tool_get_session(home: &Path, arguments: &Value) -> Result<Value, ToolError> {
    let tool = required_tool(arguments, "tool")?;
    let session_id = required_string(arguments, "session_id")?;
    let limit_events = clamped_usize(arguments, "limit_events", 100, 1, 500)?;
    let after_raw_line = optional_i64_min(arguments, "after_raw_line", 0)?;
    let around_raw_line = optional_i64_min(arguments, "around_raw_line", 1)?;
    let before = clamped_usize(arguments, "before", 5, 0, 500)?;
    let after = clamped_usize(arguments, "after", 5, 0, 500)?;
    let redact = optional_bool(arguments, "redact", false);
    Ok(serde_json::to_value(get_session_page(
        home,
        tool,
        session_id,
        SessionOptions {
            limit_events,
            after_raw_line,
            around_raw_line,
            before,
            after,
            include_deltas: optional_bool(arguments, "include_deltas", false),
            canonical_type: optional_string(arguments, "canonical_type"),
            redact,
            corroborate: optional_bool(arguments, "corroborate", false),
        },
    )?)?)
}

fn tool_export_session(home: &Path, arguments: &Value) -> Result<Value, ToolError> {
    let tool = required_tool(arguments, "tool")?;
    let session_id = required_string(arguments, "session_id")?;
    let format = optional_string(arguments, "format").unwrap_or_else(|| "markdown".to_string());
    let redact = optional_bool(arguments, "redact", false);
    let content = match format.as_str() {
        "markdown" => export_session_markdown_with_options(home, tool, session_id, redact)?,
        "jsonl" => export_session_jsonl_with_options(home, tool, session_id, redact)?,
        _ => {
            return Err(ToolError::new(
                "VALIDATION_ERROR",
                "format must be markdown or jsonl",
                true,
            ))
        }
    };
    let structured = json!({
        "tool": tool,
        "session_id": session_id,
        "format": format,
        "content": content,
        "raw_file": nabu_core::canonical_raw_path(home, tool, session_id),
        "redacted": redact
    });
    Ok(enforce_size_bound(structured, format == "jsonl"))
}

fn tool_get_event(home: &Path, arguments: &Value) -> Result<Value, ToolError> {
    let tool = required_tool(arguments, "tool")?;
    let session_id = required_string(arguments, "session_id")?;
    let raw_line = optional_i64_min(arguments, "raw_line", 1)?;
    let raw_offset = optional_i64_min(arguments, "raw_offset", 0)?;
    let redact = optional_bool(arguments, "redact", false);
    Ok(serde_json::to_value(get_event_by_pointer_with_options(
        home,
        tool,
        session_id,
        raw_line,
        raw_offset,
        EventOptions {
            redact,
            corroborate: optional_bool(arguments, "corroborate", false),
        },
    )?)?)
}

fn tool_history_doctor(home: &Path, arguments: &Value) -> Result<Value, ToolError> {
    let tool = optional_string(arguments, "tool").unwrap_or_else(|| "all".to_string());
    let deep = optional_bool(arguments, "deep", false);
    match tool.as_str() {
        "codex" | "claude" | "opencode" | "all" => {}
        _ => {
            return Err(ToolError::new(
                "VALIDATION_ERROR",
                "tool must be codex, claude, opencode, or all",
                true,
            ))
        }
    }
    if deep {
        ensure_mcp_deep_doctor_allowed(home)?;
    }
    Ok(json!({
        "ok": true,
        "data": {
            "tool": tool,
            "protocol_version": PROTOCOL_VERSION,
            "server_version": env!("CARGO_PKG_VERSION"),
            "health": doctor_with_options(home, deep)
        }
    }))
}

fn ensure_mcp_deep_doctor_allowed(home: &Path) -> Result<(), ToolError> {
    let Some(max_bytes) = mcp_deep_doctor_max_bytes() else {
        return Ok(());
    };
    let db_path = home.join("index").join("harness.db");
    let db_size_bytes = match std::fs::metadata(&db_path) {
        Ok(metadata) => metadata.len(),
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(source) => {
            return Err(Error::Io {
                path: db_path,
                source,
            }
            .into())
        }
    };
    if db_size_bytes <= max_bytes {
        return Ok(());
    }

    Err(ToolError::with_details(
        "VALIDATION_ERROR",
        format!(
            "history_doctor deep=true would run SQLite integrity_check over a {db_size_bytes}-byte index, above the MCP limit of {max_bytes} bytes"
        ),
        true,
        json!({
            "reason": "deep_doctor_index_too_large",
            "db_path": db_path.display().to_string(),
            "db_size_bytes": db_size_bytes,
            "max_bytes": max_bytes,
            "suggested_command": "nabu doctor --deep",
            "override_env": "NABU_MCP_DEEP_DOCTOR_MAX_BYTES"
        }),
    ))
}

fn tool_recall_answer(home: &Path, arguments: &Value) -> Result<Value, ToolError> {
    let query = required_string(arguments, "query")?;
    if query.trim().is_empty() {
        return Err(ToolError::new(
            "VALIDATION_ERROR",
            "query must not be empty",
            true,
        ));
    }
    let limit = clamped_usize(arguments, "limit", 5, 1, 10)?;
    let before = clamped_usize(arguments, "before", 5, 0, 20)?;
    let after = clamped_usize(arguments, "after", 5, 0, 20)?;
    let redact = optional_bool(arguments, "redact", false);
    let corroborate = optional_bool(arguments, "corroborate", false);
    let search_page = search_history_page(
        home,
        query,
        SearchOptions {
            tool: optional_tool(arguments, "tool")?,
            session_id: optional_string(arguments, "session_id"),
            cwd: optional_string(arguments, "cwd"),
            since: optional_string(arguments, "since"),
            canonical_type: optional_string(arguments, "canonical_type"),
            file: optional_string(arguments, "file"),
            command: optional_string(arguments, "command"),
            limit,
            offset: 0,
            include_payload: false,
            include_deltas: false,
            dedupe: true,
            max_snippet_chars: 240,
            mode: optional_search_mode(arguments)?,
            corroborate,
        },
    )?;

    let mut context_windows = recall_context_windows(&search_page.results, before, after);
    let mut hits = Vec::new();
    let mut response = json!({
        "query": query,
        "mode_requested": search_page.mode_requested,
        "mode_applied": search_page.mode_applied,
        "semantic_available": search_page.semantic_available,
        "hits": [],
        "returned": 0,
        "truncated": search_page.truncated,
        "redacted": redact
    });
    let Some(mut budget) = ResponseBudget::from_base(&response) else {
        return Ok(enforce_size_bound(response, false));
    };
    let mut seen_context = std::collections::BTreeSet::new();
    let mut budget_truncated = false;
    for (index, hit) in search_page.results.iter().enumerate() {
        let context = recall_context_for_hit(
            home,
            hit,
            &mut context_windows,
            RecallParams {
                before,
                after,
                redact,
                corroborate,
            },
            &mut seen_context,
        )?;
        let mut snippet = hit.snippet.clone();
        if redact {
            snippet = redact_export_text(&snippet);
        }
        let mut hit_value = json!({
            "rank": index + 1,
            "score": hit.score,
            "tool": hit.tool,
            "session_id": hit.session_id,
            "canonical_type": hit.canonical_type,
            "timestamp": hit.timestamp,
            "snippet": snippet,
            "raw_file": hit.raw_file,
            "raw_line": hit.raw_line,
            "raw_offset": hit.raw_offset,
            "also_at": hit.also_at,
            "context": context
        });
        if let Some(corroboration) = &hit.corroboration {
            hit_value["corroboration"] = serde_json::to_value(corroboration)?;
        }
        let separator = usize::from(!hits.is_empty());
        if !budget.try_reserve_value(&hit_value, separator) {
            budget_truncated = true;
            break;
        }
        hits.push(hit_value);
    }

    let returned = hits.len();
    response["hits"] = Value::Array(hits);
    response["returned"] = json!(returned);
    response["truncated"] = json!(search_page.truncated || budget_truncated);
    trim_recall_response_to_limit(&mut response);
    Ok(enforce_size_bound(response, false))
}

fn recall_context_windows(
    hits: &[SearchResult],
    before: usize,
    after: usize,
) -> std::collections::BTreeMap<(String, String), Vec<RecallWindow>> {
    let mut planned = std::collections::BTreeMap::<(String, String), Vec<(Tool, i64, i64)>>::new();
    for hit in hits {
        let start = hit.raw_line.saturating_sub(before as i64).max(1);
        let end = hit.raw_line.saturating_add(after as i64);
        planned
            .entry((hit.tool.as_str().to_string(), hit.session_id.clone()))
            .or_default()
            .push((hit.tool, start, end));
    }

    let mut fetched = std::collections::BTreeMap::new();
    for ((tool_name, session_id), mut windows) in planned {
        windows.sort_by_key(|(_, start, end)| (*start, *end));
        let mut merged = Vec::<(i64, i64)>::new();
        for (_, start, end) in windows {
            if let Some((_, current_end)) = merged.last_mut() {
                if start <= current_end.saturating_add(1) {
                    *current_end = (*current_end).max(end);
                    continue;
                }
            }
            merged.push((start, end));
        }

        fetched.insert(
            (tool_name, session_id),
            merged
                .into_iter()
                .map(|(start, end)| RecallWindow {
                    start,
                    end,
                    events: None,
                })
                .collect(),
        );
    }
    fetched
}

fn trim_recall_response_to_limit(value: &mut Value) {
    while serde_json::to_vec(value)
        .map(|serialized| serialized.len() > MAX_MCP_BYTES)
        .unwrap_or(false)
    {
        let Some(hits) = value.get_mut("hits").and_then(Value::as_array_mut) else {
            break;
        };
        if hits.pop().is_none() {
            break;
        }
        let returned = hits.len();
        value["returned"] = json!(returned);
        value["truncated"] = json!(true);
    }
}

/// Recall rendering parameters that travel together when materializing a hit's
/// surrounding context.
#[derive(Clone, Copy)]
struct RecallParams {
    before: usize,
    after: usize,
    redact: bool,
    corroborate: bool,
}

fn recall_context_for_hit(
    home: &Path,
    hit: &SearchResult,
    windows: &mut std::collections::BTreeMap<(String, String), Vec<RecallWindow>>,
    params: RecallParams,
    seen_context: &mut std::collections::BTreeSet<(String, String, i64)>,
) -> Result<Vec<Value>, ToolError> {
    let RecallParams {
        before,
        after,
        redact,
        corroborate,
    } = params;
    let start = hit.raw_line.saturating_sub(before as i64).max(1);
    let end = hit.raw_line.saturating_add(after as i64);
    let key = (hit.tool.as_str().to_string(), hit.session_id.clone());
    let Some(windows) = windows.get_mut(&key) else {
        return Ok(Vec::new());
    };
    let mut context = Vec::new();
    for window in windows {
        if end < window.start || start > window.end {
            continue;
        }
        let events = window_events(home, hit, window, redact, corroborate)?;
        for event in events {
            if event.raw_line < start || event.raw_line > end {
                continue;
            }
            let key = (
                event.tool.as_str().to_string(),
                event.session_id.clone(),
                event.raw_line,
            );
            if seen_context.insert(key) {
                context.push(serde_json::to_value(event)?);
            }
        }
    }
    Ok(context)
}

fn window_events<'a>(
    home: &Path,
    hit: &SearchResult,
    window: &'a mut RecallWindow,
    redact: bool,
    corroborate: bool,
) -> Result<&'a [StoredEvent], ToolError> {
    if window.events.is_none() {
        let center = window
            .start
            .saturating_add(window.end)
            .saturating_div(2)
            .max(1);
        let before = usize::try_from(center.saturating_sub(window.start)).unwrap_or(usize::MAX);
        let after = usize::try_from(window.end.saturating_sub(center)).unwrap_or(usize::MAX);
        let session = get_session_page(
            home,
            hit.tool,
            &hit.session_id,
            SessionOptions {
                limit_events: before.saturating_add(after).saturating_add(1),
                after_raw_line: None,
                around_raw_line: Some(center),
                before,
                after,
                include_deltas: false,
                canonical_type: None,
                redact,
                corroborate,
            },
        )?;
        window.events = Some(session.events);
    }
    Ok(window.events.as_deref().unwrap_or(&[]))
}

fn handle_resource_read(home: &Path, params: &Value) -> Value {
    let uri = params.get("uri").and_then(Value::as_str).unwrap_or("");
    let content = match resource_content(home, uri) {
        Ok(content) => content,
        Err(error) => json!({
            "error": {
                "code": "resource_read_failed",
                "message": error.to_string(),
                "recoverable": true
            }
        })
        .to_string(),
    };
    json!({
        "contents": [
            {
                "uri": uri,
                "mimeType": "application/json",
                "text": truncate_resource(content)
            }
        ]
    })
}

fn resource_content(home: &Path, uri: &str) -> nabu_core::Result<String> {
    match uri {
        "nabu://sessions" => {
            let sessions = list_sessions(home, None, None, None, 20)?;
            Ok(serde_json::to_string(&json!({ "sessions": sessions }))?)
        }
        "nabu://schema/tools" => Ok(serde_json::to_string(&json!({
            "tools": tool_descriptions()
        }))?),
        _ if uri.starts_with("nabu://sessions/") => {
            let rest = uri.trim_start_matches("nabu://sessions/");
            let mut parts = rest.splitn(2, '/');
            let tool = parts
                .next()
                .and_then(|value| Tool::from_str(value).ok())
                .ok_or_else(|| Error::Validation("resource tool is invalid".to_string()))?;
            let session_id = parts
                .next()
                .filter(|value| !value.is_empty())
                .ok_or_else(|| Error::Validation("resource session_id is required".to_string()))?;
            Ok(serde_json::to_string(&get_session_page(
                home,
                tool,
                session_id,
                SessionOptions {
                    include_deltas: true,
                    ..SessionOptions::default()
                },
            )?)?)
        }
        _ => Err(Error::Validation(format!("unknown resource uri: {uri}"))),
    }
}

fn handle_prompt_get(params: &Value) -> Value {
    let name = params.get("name").and_then(Value::as_str).unwrap_or("");
    let text = match name {
        "recall_project_history" => {
            "Before continuing, call search_history with a concise project query. It is citation-first and payload-light by default; page with offset and drill into relevant hits using get_session around_raw_line/before/after or get_event. Cite tool, session_id, and raw_line."
        }
        "prepare_handoff_summary" => {
            "Call list_sessions, then get_session with around_raw_line windows or export_session for full-fidelity handoff content. Produce a compact handoff summary with citations including tool, session_id, raw_line or raw_offset."
        }
        _ => "Unknown prompt. Call prompts/list to discover available nabu prompts.",
    };
    json!({
        "description": name,
        "messages": [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": text
                }
            }
        ]
    })
}

fn resource_descriptions() -> Value {
    json!([
        {
            "uri": "nabu://sessions",
            "name": "Recent nabu sessions",
            "description": "Recent session metadata with raw citations.",
            "mimeType": "application/json"
        },
        {
            "uriTemplate": "nabu://sessions/{tool}/{session_id}",
            "name": "nabu session summary",
            "description": "Bounded normalized events for one session.",
            "mimeType": "application/json"
        },
        {
            "uri": "nabu://schema/tools",
            "name": "nabu MCP tool schemas",
            "description": "Input schemas for all read-only history tools.",
            "mimeType": "application/json"
        }
    ])
}

fn prompt_descriptions() -> Value {
    json!([
        {
            "name": "recall_project_history",
            "description": "Search local history before continuing project work.",
            "arguments": []
        },
        {
            "name": "prepare_handoff_summary",
            "description": "Retrieve relevant sessions and prepare a cited handoff.",
            "arguments": []
        }
    ])
}

pub fn tool_schemas_document() -> Value {
    json!({
        "transport": "stdio",
        "tools": {
            "search_history": tool_schema("search_history"),
            "list_sessions": tool_schema("list_sessions"),
            "get_session": tool_schema("get_session"),
            "export_session": tool_schema("export_session"),
            "get_event": tool_schema("get_event"),
            "history_doctor": tool_schema("history_doctor"),
            "recall_answer": tool_schema("recall_answer")
        }
    })
}

fn tool_schema(name: &str) -> Value {
    match name {
        "search_history" => json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "minLength": 1 },
                "tool": { "type": "string", "enum": ["codex", "claude", "opencode"] },
                "session_id": { "type": "string" },
                "cwd": { "type": "string" },
                "since": { "type": "string" },
                "canonical_type": { "type": "string" },
                "file": { "type": "string" },
                "command": { "type": "string" },
                "mode": { "type": "string", "enum": ["auto", "lexical", "hybrid"], "default": "auto" },
                "limit": { "type": "integer", "minimum": 1, "maximum": 50, "default": 10 },
                "offset": { "type": "integer", "minimum": 0, "default": 0 },
                "include_payload": { "type": "boolean", "default": false },
                "include_deltas": { "type": "boolean", "default": false },
                "dedupe": { "type": "boolean", "default": true },
                "max_snippet_chars": { "type": "integer", "minimum": 1, "maximum": 1000, "default": 240 },
                "corroborate": { "type": "boolean", "default": false, "description": "When true, annotate results with local read-only git checks for mentioned commits, branches, and files. PR refs are unresolved with reason=needs_network; no fetch or forge call is made." },
                "redact": { "type": "boolean", "default": false }
            },
            "required": ["query"],
            "additionalProperties": false
        }),
        "list_sessions" => json!({
            "type": "object",
            "properties": {
                "tool": { "type": "string", "enum": ["codex", "claude", "opencode"] },
                "cwd": { "type": "string" },
                "since": { "type": "string" },
                "limit": { "type": "integer", "minimum": 1, "maximum": 100, "default": 20 }
            },
            "additionalProperties": false
        }),
        "get_session" => json!({
            "type": "object",
            "properties": {
                "tool": { "type": "string", "enum": ["codex", "claude", "opencode"] },
                "session_id": { "type": "string", "minLength": 1 },
                "limit_events": { "type": "integer", "minimum": 1, "maximum": 500, "default": 100 },
                "after_raw_line": { "type": "integer", "minimum": 0 },
                "around_raw_line": { "type": "integer", "minimum": 1 },
                "before": { "type": "integer", "minimum": 0, "maximum": 500, "default": 5 },
                "after": { "type": "integer", "minimum": 0, "maximum": 500, "default": 5 },
                "canonical_type": { "type": "string" },
                "include_deltas": { "type": "boolean", "default": false },
                "corroborate": { "type": "boolean", "default": false, "description": "When true, annotate events with local read-only git checks. PR refs are unresolved with reason=needs_network." },
                "redact": { "type": "boolean", "default": false }
            },
            "required": ["tool", "session_id"],
            "additionalProperties": false
        }),
        "export_session" => json!({
            "type": "object",
            "properties": {
                "tool": { "type": "string", "enum": ["codex", "claude", "opencode"] },
                "session_id": { "type": "string", "minLength": 1 },
                "format": { "type": "string", "enum": ["markdown", "jsonl"], "default": "markdown" },
                "redact": { "type": "boolean", "default": false }
            },
            "required": ["tool", "session_id"],
            "additionalProperties": false
        }),
        "get_event" => json!({
            "type": "object",
            "properties": {
                "tool": { "type": "string", "enum": ["codex", "claude", "opencode"] },
                "session_id": { "type": "string", "minLength": 1 },
                "raw_line": { "type": "integer", "minimum": 1 },
                "raw_offset": { "type": "integer", "minimum": 0 },
                "corroborate": { "type": "boolean", "default": false, "description": "When true, annotate the event with local read-only git checks. PR refs are unresolved with reason=needs_network." },
                "redact": { "type": "boolean", "default": false }
            },
            "required": ["tool", "session_id"],
            "additionalProperties": false
        }),
        "history_doctor" => json!({
            "type": "object",
            "properties": {
                "tool": { "type": "string", "enum": ["codex", "claude", "opencode", "all"], "default": "all" },
                "deep": { "type": "boolean", "default": false, "description": "Runs full SQLite integrity_check. Over MCP this is refused when harness.db exceeds NABU_MCP_DEEP_DOCTOR_MAX_BYTES (default 500 MiB; 0 disables the guard)." }
            },
            "additionalProperties": false
        }),
        "recall_answer" => json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "minLength": 1 },
                "tool": { "type": "string", "enum": ["codex", "claude", "opencode"] },
                "session_id": { "type": "string" },
                "cwd": { "type": "string" },
                "since": { "type": "string" },
                "canonical_type": { "type": "string" },
                "file": { "type": "string" },
                "command": { "type": "string" },
                "mode": { "type": "string", "enum": ["auto", "lexical", "hybrid"], "default": "auto" },
                "limit": { "type": "integer", "minimum": 1, "maximum": 10, "default": 5 },
                "before": { "type": "integer", "minimum": 0, "maximum": 20, "default": 5 },
                "after": { "type": "integer", "minimum": 0, "maximum": 20, "default": 5 },
                "corroborate": { "type": "boolean", "default": false, "description": "When true, annotate hits and context with local read-only git checks. PR refs are unresolved with reason=needs_network." },
                "redact": { "type": "boolean", "default": false }
            },
            "required": ["query"],
            "additionalProperties": false
        }),
        _ => json!({ "type": "object", "additionalProperties": false }),
    }
}

fn concise_summary(name: &str, structured: &Value) -> String {
    match name {
        "search_history" => format!(
            "Found {} history result(s){}.",
            structured
                .get("results")
                .and_then(Value::as_array)
                .map(Vec::len)
                .unwrap_or(0),
            if structured
                .get("truncated")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                " (truncated; use continuation.next_offset)"
            } else {
                ""
            }
        ),
        "list_sessions" => format!(
            "Found {} session(s).",
            structured
                .get("sessions")
                .and_then(Value::as_array)
                .map(Vec::len)
                .unwrap_or(0)
        ),
        "get_session" => format!(
            "Returned {} event(s) for {}:{}.",
            structured
                .get("events")
                .and_then(Value::as_array)
                .map(Vec::len)
                .unwrap_or(0),
            structured
                .get("tool")
                .and_then(Value::as_str)
                .unwrap_or("tool"),
            structured
                .get("session_id")
                .and_then(Value::as_str)
                .unwrap_or("session")
        ),
        "export_session" => format!(
            "Exported {}:{} as {}.",
            structured
                .get("tool")
                .and_then(Value::as_str)
                .unwrap_or("tool"),
            structured
                .get("session_id")
                .and_then(Value::as_str)
                .unwrap_or("session"),
            structured
                .get("format")
                .and_then(Value::as_str)
                .unwrap_or("markdown")
        ),
        "get_event" => "Returned one event with raw citation.".to_string(),
        "history_doctor" => "Returned nabu health checks.".to_string(),
        "recall_answer" => format!(
            "Assembled {} cited context hit(s); no answer text was generated.",
            structured
                .get("hits")
                .and_then(Value::as_array)
                .map(Vec::len)
                .unwrap_or(0)
        ),
        _ => "Completed MCP tool call.".to_string(),
    }
}

fn enforce_size_bound(value: Value, allow_jsonl: bool) -> Value {
    if allow_jsonl {
        return value;
    }
    let Ok(serialized) = serde_json::to_vec(&value) else {
        return value;
    };
    if serialized.len() <= MAX_MCP_BYTES {
        return value;
    }
    json!({
        "truncated": true,
        "message": "MCP response exceeded 256 KiB. Call get_session with pagination or export_session with format=jsonl.",
        "original_size_bytes": serialized.len()
    })
}

fn fit_search_response(mut value: Value) -> Value {
    let Ok(serialized) = serde_json::to_vec(&value) else {
        return value;
    };
    if serialized.len() <= MAX_MCP_BYTES {
        return value;
    }

    let Some(results) = value.get("results").and_then(Value::as_array).cloned() else {
        return enforce_size_bound(value, false);
    };
    let offset = value
        .get("offset_applied")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;
    value["results"] = json!([]);
    value["returned"] = json!(0);
    value["truncated"] = json!(true);
    value["continuation"] = json!({ "next_offset": offset });

    let Some(mut budget) = ResponseBudget::from_base(&value) else {
        return enforce_size_bound(value, false);
    };
    let mut kept = Vec::new();
    for result in results {
        let separator = usize::from(!kept.is_empty());
        if !budget.try_reserve_value(&result, separator) {
            break;
        }
        kept.push(result);
    }

    value["results"] = Value::Array(kept);
    let returned = value
        .get("results")
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or(0);
    value["returned"] = json!(returned);
    value["continuation"] = json!({ "next_offset": offset.saturating_add(returned) });
    trim_search_response_to_limit(&mut value, offset);
    value
}

fn trim_search_response_to_limit(value: &mut Value, offset: usize) {
    while serde_json::to_vec(value)
        .map(|serialized| serialized.len() > MAX_MCP_BYTES)
        .unwrap_or(false)
    {
        let Some(results) = value.get_mut("results").and_then(Value::as_array_mut) else {
            break;
        };
        if results.pop().is_none() {
            break;
        }
        let returned = results.len();
        value["returned"] = json!(returned);
        value["continuation"] = json!({ "next_offset": offset.saturating_add(returned) });
    }
}

fn truncate_resource(content: String) -> String {
    if content.len() <= MAX_MCP_BYTES {
        content
    } else {
        json!({
            "truncated": true,
            "message": "Resource exceeded 256 KiB. Call get_session with pagination or export_session for full content."
        })
        .to_string()
    }
}

fn required_string<'a>(arguments: &'a Value, key: &str) -> Result<&'a str, ToolError> {
    arguments
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| ToolError::new("VALIDATION_ERROR", format!("{key} is required"), true))
}

fn optional_string(arguments: &Value, key: &str) -> Option<String> {
    arguments
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn required_tool(arguments: &Value, key: &str) -> Result<Tool, ToolError> {
    let value = required_string(arguments, key)?;
    Tool::from_str(value).map_err(|_| {
        ToolError::new(
            "VALIDATION_ERROR",
            format!("{key} must be codex, claude, or opencode"),
            true,
        )
    })
}

fn optional_tool(arguments: &Value, key: &str) -> Result<Option<Tool>, ToolError> {
    let Some(value) = arguments.get(key).and_then(Value::as_str) else {
        return Ok(None);
    };
    Tool::from_str(value).map(Some).map_err(|_| {
        ToolError::new(
            "VALIDATION_ERROR",
            format!("{key} must be codex, claude, or opencode"),
            true,
        )
    })
}

fn optional_search_mode(arguments: &Value) -> Result<SearchMode, ToolError> {
    let Some(value) = arguments.get("mode").and_then(Value::as_str) else {
        return Ok(SearchMode::Auto);
    };
    SearchMode::from_str(value).map_err(|_| {
        ToolError::new(
            "VALIDATION_ERROR",
            "mode must be auto, lexical, or hybrid",
            true,
        )
    })
}

fn optional_bool(arguments: &Value, key: &str, default: bool) -> bool {
    arguments
        .get(key)
        .and_then(Value::as_bool)
        .unwrap_or(default)
}

fn optional_i64_min(arguments: &Value, key: &str, min: i64) -> Result<Option<i64>, ToolError> {
    let Some(value) = arguments.get(key) else {
        return Ok(None);
    };
    let Some(value) = value.as_i64() else {
        return Err(ToolError::new(
            "VALIDATION_ERROR",
            format!("{key} must be an integer"),
            true,
        ));
    };
    if value < min {
        return Err(ToolError::new(
            "VALIDATION_ERROR",
            format!("{key} must be at least {min}"),
            true,
        ));
    }
    Ok(Some(value))
}

fn bounded_usize(
    arguments: &Value,
    key: &str,
    default: usize,
    min: usize,
    max: usize,
) -> Result<usize, ToolError> {
    let value = arguments
        .get(key)
        .and_then(Value::as_i64)
        .map(|value| value as isize)
        .unwrap_or(default as isize);
    if value < min as isize || value > max as isize {
        return Err(ToolError::new(
            "VALIDATION_ERROR",
            format!("{key} must be between {min} and {max}"),
            true,
        ));
    }
    Ok(value as usize)
}

fn clamped_usize(
    arguments: &Value,
    key: &str,
    default: usize,
    min: usize,
    max: usize,
) -> Result<usize, ToolError> {
    let value = arguments
        .get(key)
        .and_then(Value::as_i64)
        .map(|value| value as isize)
        .unwrap_or(default as isize);
    if value < min as isize {
        return Err(ToolError::new(
            "VALIDATION_ERROR",
            format!("{key} must be at least {min}"),
            true,
        ));
    }
    Ok((value as usize).min(max))
}

fn optional_usize_min(
    arguments: &Value,
    key: &str,
    min: usize,
) -> Result<Option<usize>, ToolError> {
    let Some(value) = arguments.get(key) else {
        return Ok(None);
    };
    let Some(value) = value.as_i64() else {
        return Err(ToolError::new(
            "VALIDATION_ERROR",
            format!("{key} must be an integer"),
            true,
        ));
    };
    if value < min as i64 {
        return Err(ToolError::new(
            "VALIDATION_ERROR",
            format!("{key} must be at least {min}"),
            true,
        ));
    }
    Ok(Some(value as usize))
}

fn redact_result_snippets(value: &mut Value) {
    let Some(results) = value.get_mut("results").and_then(Value::as_array_mut) else {
        return;
    };
    for result in results {
        if let Some(snippet) = result.get("snippet").and_then(Value::as_str) {
            let redacted = redact_export_text(snippet);
            result["snippet"] = Value::String(redacted);
        }
        if let Some(payload) = result.get_mut("payload") {
            *payload = redact_export_json(std::mem::take(payload));
        }
    }
}

#[derive(Debug)]
struct ToolError {
    code: &'static str,
    message: String,
    recoverable: bool,
    hint: String,
    details: Value,
}

impl ToolError {
    fn new(code: &'static str, message: impl Into<String>, recoverable: bool) -> Self {
        let message = message.into();
        Self {
            code,
            hint: hint_for_code(code).to_string(),
            details: json!({}),
            message,
            recoverable,
        }
    }

    fn with_details(
        code: &'static str,
        message: impl Into<String>,
        recoverable: bool,
        details: Value,
    ) -> Self {
        let message = message.into();
        Self {
            code,
            hint: hint_for_code(code).to_string(),
            details,
            message,
            recoverable,
        }
    }
}

impl std::fmt::Display for ToolError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl From<nabu_core::Error> for ToolError {
    fn from(value: nabu_core::Error) -> Self {
        let message = value.to_string();
        let code = if message.contains("not found") {
            "NOT_FOUND"
        } else {
            match &value {
                nabu_core::Error::Validation(_) => "VALIDATION_ERROR",
                nabu_core::Error::SemanticUnavailable(_) => "SEMANTIC_UNAVAILABLE",
                nabu_core::Error::HomeUnavailable => "STORAGE_UNAVAILABLE",
                nabu_core::Error::Io { source, .. }
                    if source.kind() == std::io::ErrorKind::PermissionDenied =>
                {
                    "PERMISSION_DENIED"
                }
                nabu_core::Error::Io { .. } => "STORAGE_UNAVAILABLE",
                nabu_core::Error::Sqlite { .. } => "INDEX_UNAVAILABLE",
                nabu_core::Error::Json(_) => "VALIDATION_ERROR",
                nabu_core::Error::TimeFormat(_) => "INTERNAL_ERROR",
            }
        };
        let details = match value {
            nabu_core::Error::Io { path, ref source }
                if source.kind() == std::io::ErrorKind::PermissionDenied =>
            {
                json!({
                    "path": path.display().to_string(),
                    "attempted_operation": "filesystem access"
                })
            }
            _ => json!({}),
        };
        Self::with_details(code, message, true, details)
    }
}

impl From<serde_json::Error> for ToolError {
    fn from(value: serde_json::Error) -> Self {
        Self::new("VALIDATION_ERROR", value.to_string(), true)
    }
}

fn hint_for_code(code: &str) -> &'static str {
    match code {
        "NOT_FOUND" => "Call list_sessions first, then retry with an existing raw pointer.",
        "VALIDATION_ERROR" => "Fix the MCP tool arguments and retry.",
        "PERMISSION_DENIED" => "Check ownership and filesystem permissions for the reported path.",
        "INDEX_UNAVAILABLE" => "Run nabu index --once and retry.",
        "STORAGE_UNAVAILABLE" => "Run nabu init and check local filesystem permissions.",
        "SEMANTIC_UNAVAILABLE" => {
            "Retry with mode=lexical, or install a compatible semantic build and local model."
        }
        _ => "Retry after checking nabu doctor.",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nabu_core::{index_once, ingest_hook_event, init_home};
    use serde_json::json;
    use std::io::Cursor;
    use tempfile::tempdir;

    #[test]
    fn stdio_initialize_lists_locked_tools() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();
        let input = Cursor::new(
            "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{}}\n\
             {\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/list\",\"params\":{}}\n",
        );
        let mut output = Vec::new();

        serve_with_io(home, input, &mut output).unwrap();
        let output = String::from_utf8(output).unwrap();

        assert!(output.contains("\"protocolVersion\":\"2025-03-26\""));
        for tool in [
            "search_history",
            "list_sessions",
            "get_session",
            "export_session",
            "get_event",
            "history_doctor",
            "recall_answer",
        ] {
            assert!(output.contains(tool), "{tool}");
        }
    }

    #[test]
    fn search_history_matches_indexed_fixture() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        init_home(&home).unwrap();
        ingest_hook_event(
            &home,
            Tool::Claude,
            json!({
                "session_id": "fixture-session",
                "hook_event_name": "UserPromptSubmit",
                "message_id": "mcp-user-1",
                "cwd": "/tmp/nabu-fixture",
                "project_root": "/tmp/nabu-fixture",
                "prompt": "nabu fixture marker for mcp search"
            }),
        )
        .unwrap();
        index_once(&home).unwrap();

        let response = handle_message(
            &home,
            &json!({
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "search_history",
                    "arguments": {
                        "query": "nabu fixture marker",
                        "limit": 10
                    }
                }
            })
            .to_string(),
        )
        .unwrap()
        .unwrap();

        assert_eq!(response["jsonrpc"], "2.0");
        assert!(response["result"]["structuredContent"]["results"]
            .as_array()
            .unwrap()
            .iter()
            .any(|result| result["session_id"] == "fixture-session"));
    }

    #[test]
    fn history_doctor_deep_rejects_large_index_before_integrity_check() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        let db_path = home.join("index").join("harness.db");
        std::fs::create_dir_all(db_path.parent().unwrap()).unwrap();
        std::fs::File::create(&db_path)
            .unwrap()
            .set_len(DEFAULT_MCP_DEEP_DOCTOR_MAX_BYTES + 1)
            .unwrap();

        let response = handle_message(
            &home,
            &json!({
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "history_doctor",
                    "arguments": {
                        "deep": true
                    }
                }
            })
            .to_string(),
        )
        .unwrap()
        .unwrap();

        let error = &response["result"]["structuredContent"]["error"];
        assert_eq!(response["result"]["isError"], true);
        assert_eq!(error["code"], "VALIDATION_ERROR");
        assert!(error["message"]
            .as_str()
            .unwrap()
            .contains("integrity_check"));
        assert_eq!(error["details"]["reason"], "deep_doctor_index_too_large");
        assert_eq!(
            error["details"]["db_size_bytes"],
            DEFAULT_MCP_DEEP_DOCTOR_MAX_BYTES + 1
        );
        assert_eq!(
            error["details"]["max_bytes"],
            DEFAULT_MCP_DEEP_DOCTOR_MAX_BYTES
        );
        assert_eq!(error["details"]["suggested_command"], "nabu doctor --deep");
    }

    #[test]
    fn permission_denied_errors_include_recovery_details() {
        let error = ToolError::from(nabu_core::Error::Io {
            path: PathBuf::from("/tmp/nabu-denied"),
            source: std::io::Error::from(std::io::ErrorKind::PermissionDenied),
        });

        assert_eq!(error.code, "PERMISSION_DENIED");
        assert!(error.recoverable);
        assert_eq!(error.details["path"], "/tmp/nabu-denied");
        assert_eq!(error.details["attempted_operation"], "filesystem access");
    }
}
