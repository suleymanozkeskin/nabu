# Capture Guarantees

## Compaction Survival

Compaction shrinks an agent's live context window. It does not shrink nabu's
record. Capture is independent of the window: every turn is written to the
append-only raw store as it happens, and the `PreCompact`/`PostCompact` hooks
fire at the compaction boundary, so the event sequence is recorded on both sides
of it. The raw JSONL is append-only and is never rewritten by compaction.

A turn that is summarized out of the live window therefore stays retrievable
verbatim — addressed by `tool:session:raw_line` — through `search`, `show`,
`export`, and the MCP read surfaces.

The pre-compaction turns were already captured live, so when the `PreCompact`
hook runs nabu dedupes them by event identity rather than rewriting them (see
Event Identity); the boundary is recorded once, not duplicated, which is why the
`PreCompact` ingest reports `skipped duplicate` while `PostCompact` appends the
new boundary. Claude Code emits `compaction.before`/`compaction.after` events;
Codex compatibility mode captures compaction boundaries through its hook events.

## Codex Compatibility Mode

Codex compatibility mode captures supported hook events at session, prompt, tool, compaction, subagent, and stop boundaries. It also reconciles local transcript files from `$CODEX_HOME/sessions` and `$CODEX_HOME/archived_sessions` when backfill is run.

This mode is not guaranteed to capture assistant deltas as they appear. For exact append-as-message-appears capture, the recommended Codex path is streaming ingestion from `codex exec --json` or app-server notifications, recorded with raw event source `exec_json` or `app_server`.

Raw payloads are preserved before normalization. If a transcript shape changes, ingestion must keep the raw payload and index a best-effort canonical event or an `error` event instead of crashing.

## Per-Tool Capture Modes

- Claude Code: live hooks are the primary capture path and include assistant display deltas when the installed Claude Code version emits `MessageDisplay`. Local transcript backfill is the reconciliation path.
- OpenCode: the user-level plugin is the primary live capture path for message, message part, session, tool, command, and file events. Server reconciliation through `GET /session/:id/message` is gap-filling only and runs only when explicitly configured.
- Codex: compatibility hooks capture turn-boundary activity and tool/session events. Exact assistant-delta capture uses `nabu ingest file --tool codex --source exec_json --path <codex-exec-jsonl>` or `--source app_server --path <notifications-jsonl>`.

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
