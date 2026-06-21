# Release Notes

## nabu 0.1.0 𒀭𒀝

First public release. nabu is a local-first, device-only, cross-agent history
keeper for coding agents.

### What nabu is

nabu captures session history from Codex, Claude Code, and OpenCode into
append-only per-session JSONL files, which are the authoritative source of truth.
A SQLite FTS5 index, fully rebuildable from the raw JSONL, serves citation-first
search to the CLI and to agents over a local MCP server. Every search result and
every retrieved event carries `tool`, `session_id`, `raw_file`, and `raw_line` or
`raw_offset`, so any agent can reuse prior work without scraping raw files or
parsing human CLI output.

### Included

- Rust CLI binary and package name: `nabu`.
- macOS and Linux support.
- Full-fidelity raw JSONL storage with one canonical raw file per tool/session
  (`codex_<session_id>.jsonl`, `claude_<session_id>.jsonl`,
  `opencode_<session_id>.jsonl`) under `~/.nabu/raw/`.
- Rebuildable SQLite FTS5 index. Deleting `index/harness.db` and rebuilding from
  raw JSONL reproduces equivalent event, message, tool-event, and compaction
  counts and equivalent search results.
- Live adapter installation for Claude Code (hooks), OpenCode (user-level
  plugin), and Codex (compatibility hooks).
- Backfill from local Codex, Claude Code, and OpenCode native-style history.
- Codex exact streaming ingestion from `exec_json` and `app_server` source files.
- Agent-facing `search`, `show`, `tail`, `export`, `doctor`, `purge`, `backfill`,
  `index`, and `bench` commands, plus an interactive `wizard` for guided
  onboarding.
- Local stdio MCP server exposing the locked MVP tool/resource/prompt surface:
  tools `search_history`, `list_sessions`, `get_session`, `export_session`,
  `get_event`, `history_doctor`, `recall_answer`; resources `nabu://sessions`,
  `nabu://sessions/{tool}/{session_id}`, `nabu://schema/tools`; prompts
  `recall_project_history`, `prepare_handoff_summary`.
- MCP install/uninstall/validate for Codex, Claude Code, and OpenCode.
- Citation-first search defaults: results include citations, score, and bounded
  snippets with `payload: null` unless `--full` or MCP `include_payload=true` is
  requested.
- Delta-light session/search defaults: `assistant.delta` rows are hidden unless
  `--include-deltas` or MCP `include_deltas=true` is requested; export remains
  full fidelity.
- Fast-by-default doctor checks with `--deep` / `deep=true` for full SQLite
  integrity and counts.
- Opt-in local read-only git corroboration (`--corroborate` / `corroborate=true`)
  that annotates results for mentioned commits, branches, and files without
  changing ranking, identity, or default output.

### Capture guarantees

- Event-identity dedupe. nabu dedupes by event identity, not by observation
  route. Native event/message/part IDs are preferred; without a native ID,
  identity uses meaningful canonical content plus native sequence/index when
  available, and content-only fallback otherwise. Hook retry and backfill
  duplicates collapse; the same logical event captured from a different route or
  at a different capture time does not produce a second raw or indexed event.
- Full-fidelity raw is the source of truth. Raw payloads are preserved before
  normalization. If an upstream transcript shape changes, ingestion keeps the raw
  payload and indexes a best-effort canonical event or an `error` event rather
  than crashing.
- The index is rebuildable from raw. All storage outside `raw/` and
  `blobs/sha256/` is derived/cache state. Full payload hydration reads canonical
  raw JSONL plus referenced blobs, not a second long-lived copy in SQLite.
- Fail-open capture. Hooks append directly to the canonical session JSONL under a
  per-session lock and return success. Capture failure never blocks agent work.

Per-tool capture modes:

- Claude Code: live hooks are the primary path, including assistant display
  deltas/finals (when the installed version emits `MessageDisplay`), tool
  calls/results, compaction boundaries, and session lifecycle events. Transcript
  backfill is the reconciliation path.
- OpenCode: the user-level plugin is the primary live path for message, message
  part, session, tool, command, and file events. Server reconciliation through
  `GET /session/:id/message` is gap-filling only, network-disabled by default,
  and runs only when `NABU_OPENCODE_URL` or `[opencode] server_url` is set.
- Codex: compatibility hooks capture turn/session/tool-boundary activity plus
  transcript reconciliation. Exact assistant-delta capture requires the streaming
  lane: `nabu ingest file --tool codex --source exec_json` or `--source
  app_server`.

See `docs/capture-guarantees.md` for the full event-identity and per-mode detail.

### Privacy and local-first posture

- Device-local storage only under `~/.nabu`.
- No product telemetry.
- No cloud, no account system, and no central nabu-controlled database.
- No network by default. No MVP command sends transcript data over the network;
  the stdio MCP server opens no network listener. OpenCode server reconciliation
  is the only network path and is off unless explicitly configured.
- Raw files preserve full local fidelity and may contain secrets. nabu does not
  redact at capture or index time.
- `export --redact` (and MCP `redact=true`) applies redaction for sharing,
  removing API keys, bearer tokens, private-key blocks, and `.env` values.
  Redaction is opt-in and never modifies raw files; it is a sharing aid, not a
  substitute for secret rotation.
- `purge --session TOOL:SESSION_ID` and `purge --before DATE_OR_DURATION` delete
  local history.
- Strict filesystem permissions: directories `0700`, files `0600`.
- Config mutations (hook/plugin/MCP install) require a dry-run preview and write a
  timestamped backup before any change.

### Known limitations

- Semantic/hybrid search is opt-in behind a default-OFF `semantic` Cargo feature
  and requires an explicitly installed local embedding model plus a built vector
  index. In the default build, `mode=auto` applies lexical BM25 and `mode=hybrid`
  returns `SEMANTIC_UNAVAILABLE`. No default path downloads a model or makes an
  embedding-related network request; model acquisition is an explicit `nabu embed
  download --model embeddinggemma-300m-q4 --yes`. Embedding is CPU-bound and slow
  on large histories. Embedding performance work is tracked as future M9.
- macOS and Linux only. No Windows support.
- The Codex transcript schema is not a stable hook interface. Compatibility-mode
  transcript reconciliation is best-effort and tolerates schema drift; exact
  Codex capture requires the streaming lane.
- The MCP server is read-only and stdio-only. MCP tools never mutate raw history,
  indexes, native agent configs, or upstream sessions. Streamable HTTP, SSE, and
  WebSocket transports are post-MVP.
- Claude Code MCP install uses the native Claude CLI when available and otherwise
  writes the equivalent user-scoped `~/.claude.json` MCP entry; the command
  reports which strategy it used.
- Search ranking is local SQLite FTS5/BM25 plus recency unless the optional
  semantic build is enabled and provisioned.
- The `trees.software` UI is roadmap-only and not part of this release.

### Performance

Verified on this machine with default features (lexical FTS5, no semantic build).
Stated as conditions → result against the Phase-8 release bar.

- Ingest (`nabu bench ingest`, direct canonical JSONL append): p95 ≈ 3.0 ms,
  p99 ≈ 3.1 ms. Bar: p95 < 50 ms, p99 < 250 ms.
- Search (`nabu bench search`): p95 ≈ 2.1 ms, p99 ≈ 4.6 ms. Bar: p95 < 200 ms.

### Consumer migration

CLI scripts and MCP callers that read `payload` from search output must pass
`nabu search --full` or MCP `search_history` with `include_payload=true` to
restore the prior populated-payload behavior. The `payload` key remains present
for structural compatibility and is `null` by default.
