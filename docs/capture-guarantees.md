# Capture Guarantees

## Codex Compatibility Mode

Codex compatibility mode captures supported hook events at session, prompt, tool, compaction, subagent, and stop boundaries.

At each `Stop` hook, nabu starts `nabu ingest codex-rollout` in a detached process for the session rollout named in `transcript_path`. At each `SubagentStop` hook, it does the same for the subagent rollout in `agent_transcript_path`. The process imports the rollout lines that no earlier pass read, then indexes them. This captures:

- every assistant message of the turn (`commentary` and `final_answer`) as `assistant.message`;
- the outer `exec` tool calls and their output;
- the commands, web searches, patches, and image reads that `exec` ran (`event_msg` / `item_completed`), as `tool.result`. Their id is the hook `tool_use_id`, so search merges them with the hook twin.

The import uses the same per-file checkpoint as `backfill`. A last line that Codex is still writing is left for the next pass. The rollout path must be absolute, and its file name must carry the thread id the hook reported; any other path is refused with a message on stderr. Backfill of `$CODEX_HOME/sessions` and `$CODEX_HOME/archived_sessions` reads the same files for sessions that ran before the hooks were installed.

Codex guardian review sessions run without hooks. Doctor counts them as `hookless_sessions`; only a backfill captures them.

`nabu install codex` writes Codex matcher groups with nested command handlers. Run `/hooks` in Codex to review and trust new or changed nabu hooks. Codex does not run an untrusted hook definition.

This mode is not guaranteed to capture assistant deltas as they appear. For exact append-as-message-appears capture, the recommended Codex path is streaming ingestion from `codex exec --json` or app-server notifications, recorded with raw event source `exec_json` or `app_server`.

Raw payloads are preserved before normalization. If a transcript shape changes, ingestion must keep the raw payload and index a best-effort canonical event instead of crashing. A record that no canonical type models (token counters, turn bookkeeping, developer instructions, or a record kind a newer tool version added) is stored as `unclassified`. `error` is reserved for real failures: an unparseable line, a broken session header, or a tool-reported error event. Events captured before `unclassified` existed keep their stored `error` type.

## Per-Tool Capture Modes

- Claude Code: live hooks are the primary capture path and include assistant display deltas when the installed Claude Code version emits `MessageDisplay`. Local transcript backfill is the reconciliation path.
- OpenCode: the user-level plugin is the primary live capture path for message, message part, session, tool, command, and file events. Server reconciliation through `GET /session/:id/message` is gap-filling only and runs only when explicitly configured.
- Codex: compatibility hooks capture turn-boundary activity and tool/session events, and each `Stop` imports the turn's rollout lines (assistant messages, tool calls, and tool output). Exact assistant-delta capture uses `nabu ingest file --tool codex --source exec_json --path <codex-exec-jsonl>` or `--source app_server --path <notifications-jsonl>`.

OpenCode server reconciliation is network-disabled by default. If neither `NABU_OPENCODE_URL` nor `[opencode] server_url` in the local nabu `config.toml` is set, `backfill --tool opencode` makes no OpenCode HTTP request. If a configured server is unavailable or returns a non-2xx response, reconciliation logs a warning and continues with local capture/backfill results.

## Event Identity

nabu dedupes by event identity, not by observation route. Native event or message IDs are preferred. Without a native ID, identity uses meaningful canonical content plus native sequence/index when available.

If an event has no native ID and no stable sequence/index, the fallback is content-only for that canonical event type. This keeps hook retry and backfill duplicates clean, but two truly identical unsequenced events may collapse. Native transcripts remain the authoritative disambiguation source when an upstream tool provides stable order.

Native ordering coverage:

- Claude: `MessageDisplay.index` is used for assistant deltas/finals. Claude transcript backfill uses the transcript byte offset when no richer payload order exists.
- Codex: hook, `exec --json`, and app-server payloads use source-provided `sequence`, `index`, `ordinal`, item order, turn order, response order, or output order when present. Codex transcript backfill uses the transcript byte offset when no richer payload order exists.
- OpenCode: `message.part.updated` and `message.part.removed` use source-provided part `index` or `sequence` when present, so a shared `message_id` does not collapse distinct part updates. OpenCode backfill uses the source byte offset when no richer payload order exists.

Cross-version caveat: events that were captured before this ordering coverage existed may have content-only keys. If the same event is later backfilled with a newly populated sequence, it can appear as a residual duplicate. This is forward-only and deterministic; nabu does not add capture-time counters to hide it because capture time is not event identity.

## Native Backfill Roots

When `backfill` is run without `--path`, nabu scans the local native roots for the selected tool: `$CODEX_HOME/sessions`, `$CODEX_HOME/archived_sessions`, `$CLAUDE_CONFIG_DIR/projects` or `~/.claude/projects`, and `~/.local/share/opencode/`.

## Semantic Retrieval

Lexical BM25 search is always available and remains the default fallback. Semantic
retrieval is additive: it is used only when a compatible `semantic` build, an
explicitly installed local model, and a built vector index are all present.

No default path downloads a model or makes an embedding-related network request.
Capture, append, index, search, doctor, export, and MCP reads must not auto-fetch
models. Model acquisition is an explicit `nabu embed download --model
embeddinggemma-300m-q4 --yes` action or an equivalent interactive wizard consent
step.

Vectors are derived index state. They live only in `index/harness.db`, are never
written to raw JSONL, and can be rebuilt from canonical raw JSONL plus referenced
blobs. Deleting `index/harness.db` removes FTS and vector state, not canonical
history.

The default build keeps semantic search unavailable unless the default-off
`semantic` feature is enabled and the local model cache/vector index are present.
When the feature, model, or vector index is absent, `mode=auto` applies lexical
search and `mode=hybrid` returns `SEMANTIC_UNAVAILABLE`.

## Source Corroboration

Corroboration is opt-in on read surfaces. `search --corroborate`,
`show --corroborate`, and MCP `corroborate=true` annotate existing results with
local git checks for mentioned commits, branches, and files. Corroboration never
changes capture, event identity, ranking, filtering, raw files, or index rows.

The git access is read-only and local. It uses local repository plumbing only,
does not run fetch/pull/remote queries, does not call forge APIs, and degrades to
`unresolved` or `unknown` annotations when the result has no repo, refs are gone,
or git reports an error.

Pull request references such as `#123` and `/pull/123` cannot be resolved under
the no-network-by-default guarantee. They are extracted and reported as
`unresolved` with `reason=needs_network`; they are never silently dropped and
never fetched by default.
