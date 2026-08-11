# Changelog

All notable user-facing changes should be recorded here. Release narratives live
in `docs/release-notes.md`.

## Unreleased

- Serve each tool's own memory folders as first-class, searchable history.
  `nabu memory sync` (also run by `nabu index --once` and the wizard; not by
  `index --watch` ticks or hook single-flight passes) captures claude
  `projects/<id>/memory/` and codex `memories/` files into the raw store as
  `memory.file` events — content-addressed, so an unchanged file never
  appends a duplicate. Nested files keep root-relative names; binary and
  >1 MiB files are skipped. Memory is searchable through `nabu search` and
  the MCP `search_history`/`recall_answer` (hits carry
  `canonical_type=memory.file` with session/raw-line citations), listed and
  read through new CLI (`nabu memory list|show`) and MCP (`list_memories`,
  `get_memory`, `nabu://memories` resources) surfaces with full content
  hydration from the raw store and optional redaction. Memory lives in
  reserved pseudo-sessions (`memory:{project}`, `memory:global`) that stay
  out of `list_sessions`; opencode has no native memory folder and
  contributes nothing. Existing indexes migrate on first open (events table
  rebuilt once, preserving row ids and restoring the full events index set).

- Add a retrieval advisory to weak lexical-only search pages. When a query
  runs lexical-only (semantic unavailable, `expand_concepts` off) and returns
  nothing or top hits scattered across sessions, the search response carries
  a one-sentence `advisory` field suggesting distinctive literal tokens or
  `expand_concepts=true`. Absent otherwise; hybrid and expanded pages never
  carry it. Surfaces in `search_history`, `recall_answer`, and the CLI
  search output.

## 0.1.4

- Stop indexing the assistant-message copy that Claude's `Stop` hook attaches
  to each turn's `session.ended` event. `last_assistant_message` and
  `transcript_path` no longer enter the session.ended document: the message is
  already indexed as the adjacent `assistant.message` event, so the copy
  doubled index and embedding size for those rows and made `get_session`
  windows spanning a turn end return the same content twice. Raw capture is
  unchanged; the full payload stays readable via `get_event`/`include_payload`.
- Collapse a tool call and its result(s) into one search slot. Search-time
  dedupe now groups `tool.call`/`tool.result` events that share a
  harness-assigned invocation id (codex `call_id`, claude `tool_use_id`),
  recorded at index time in a new `events.tool_invocation_id` column, so one
  invocation no longer fills several ranked slots as call/result twins.
  Collapsed events stay cited in `also_at`; `dedupe=false` still restores
  every row. Rows indexed before the column existed keep per-event slots.
- State the query strategy in the MCP tool descriptions: `search_history` and
  `recall_answer` now say that terms are OR-joined — distinctive literal
  tokens sharpen ranking without erasing recall — and point concept-worded
  queries at `expand_concepts=true`.

## 0.1.3

Hardening release: closes defects found in a full-codebase audit. No new
features.

- Hook install/uninstall safety (Claude Code, Codex, OpenCode):
  - Shell-quote the `--home` path in generated hook commands; a home path
    containing a space no longer silently breaks every capture.
  - Refuse valid-but-unexpected JSON shapes (array root, non-object `hooks`,
    non-array event entry) with an error instead of overwriting user config.
  - Handle multi-hook Claude entries: dedupe, status, and uninstall inspect
    every inner hook; uninstall removes only nabu's hook and keeps co-located
    user hooks; reinstall replaces stale nabu hooks (old command format or a
    changed `--home`) instead of stacking duplicates.
  - Uninstall prunes only the hook events nabu itself emptied; user-created
    empty entries survive, and uninstall on a never-installed config is a
    no-op.
  - `chmod 0o700` applies only to directories nabu itself created, never to
    pre-existing user directories.
  - All config writes are atomic (temp file + rename); a crash mid-write can
    no longer truncate a live agent config.
  - Install/uninstall reports list only the hook entries nabu touched instead
    of printing the full settings file (which can contain API keys).
  - Status and doctor report a config parse error instead of failing outright.
- MCP registration (`nabu mcp install/uninstall/validate`):
  - Codex: a `[mcp_servers.nabu]` header with a trailing comment no longer
    produces a duplicate table (invalid TOML) on reinstall.
  - OpenCode: uninstall no longer leaves a dangling comma when a comment
    precedes the nabu entry.
  - The Claude native path verifies what `claude mcp add/remove` actually did
    and reports failures instead of unconditional success.
  - Dry-run computes `changed` from the real config instead of always `true`.
  - `nabu mcp validate` probes a throwaway temporary home instead of a
    compile-time fixture path; it no longer reports `server_unhealthy` for
    healthy installs and never creates or migrates the user's real index.
- Purge completeness: `purge --session` / `--before` now also remove embedded
  unit plaintext (`vector_unit_texts`), orphaned vector embeddings, and spilled
  >16MB payload blobs no longer referenced by surviving history. The
  `purge --all` preview labels blobs as irreversible — they are authoritative
  payload, not rebuildable derived data.
- Redaction: `export --redact` now catches `export VAR=secret` and mid-line
  assignments (the rule previously anchored at line start and missed both),
  and the JSONL export applies key-based redaction (`"api_key": …`) per line.
- Semantic model download: pinned to a repository revision with per-file
  sha256 verification; files are installed via temp+verify+rename, so a
  truncated or tampered download can never be silently loaded.
- Search: hybrid mode now honors `--corroborate` and reports an exact total
  only when the result set is complete; both were silently dropped whenever
  the semantic model was installed.
- Indexing: incremental passes resume from the checkpoint and parse only the
  appended tail. Previously every hook-triggered pass re-read and re-rendered
  the whole session file, making per-session cost quadratic in session length.
- Capture fidelity: malformed hook stdin is preserved as a synthetic
  `parse_error` event (matching backfill behavior) instead of being dropped.
- `tail --follow` recovers from file truncation or rotation instead of
  streaming nothing forever.
- `NABU_MCP_MAX_CONCURRENCY` is clamped to 1..=64.
- The wizard summary shows whether history is searchable ("Searchable now:
  N matching events") instead of discarding the probe result.
- Docs: the `nabu search` flag table gains the missing `--ref` row.

## 0.1.2

- Index captured events automatically: each capture hook now triggers a
  detached, single-flight incremental index of the new delta, so
  `search_history` / `list_sessions` see sessions without a manual
  `nabu index --once`. Capture stays non-blocking — indexing runs in a spawned
  child, never inline on the hook path.
- Add an index-freshness signal to `history_doctor` and `nabu doctor`: per tool,
  `raw_bytes` / `indexed_bytes` / `unindexed_bytes` plus a `stale` flag, so a
  lagging index fails loudly instead of being masked by the index's own
  timestamp. Freshness is measured by byte offsets, not clocks, so it is not
  fooled by backfilled history whose events keep their original timestamps.

## 0.1.1

- Make the `nabu wizard` capture/backfill/connect checklists Enter-driven:
  ↑/↓ move, Enter toggles the row under the cursor, and a leading `Continue`
  row commits. Previously these used a Space-to-toggle multi-select where
  pressing Enter silently committed every pre-checked agent.

## 0.1.0

- Initial public release. See `docs/release-notes.md`.
