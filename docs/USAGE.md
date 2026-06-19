# nabu — Usage Manual

Full command reference. For what nabu is and why, see the
[README](../README.md).

`nabu` is the binary. Every command accepts a global `--home PATH`
(default `~/.nabu`; override also via `NABU_HOME`).

## Storage

Default home:

```shell
~/.nabu
```

Override per command:

```shell
nabu --home PATH <command>
```

Raw session files are canonical:

```text
raw/{tool}/{tool}_{sanitized_session_id}.jsonl
```

The SQLite index is derived and can be rebuilt from raw files.
`index/`, `spool/`, and `models/` are derived/cache state. `raw/` plus
`blobs/sha256/` are the canonical full-fidelity history store.

## Commands

Every command accepts the global `--home PATH` (default `~/.nabu`, also
settable via `NABU_HOME`). It is omitted from each flag table below.

`--since` / `--before` accept a relative duration (`Nd`, `Nh`, `Nm`, `Ns` — for
example `7d`, `24h`), a `YYYY-MM-DD` date (interpreted as UTC midnight), or an
RFC3339 timestamp.

The `<TOOL>` argument and `--tool` option are `codex`, `claude`, or `opencode`.
Where an enum also accepts `all` it is noted in the relevant table.

### `nabu init` — create the storage layout

Initialize the store directory tree under the home path. Idempotent; safe to
re-run.

```shell
nabu init [--home PATH]
```

No flags beyond `--home`.

```shell
nabu init
nabu --home /tmp/nabu-test init
```

### `nabu wizard` — guided first-run and management front end

```shell
nabu wizard [--home PATH]
```

No flags beyond `--home`.

```shell
nabu wizard
```

`wizard` is an interactive front end over the explicit commands below; it adds
no capability of its own. It detects which agents you have, previews each change,
asks before every mutation, and routes every config write through the same
`install`/`mcp install`/`backfill` functions (diff preview → consent →
timestamped backup). The explicit commands remain fully supported — the wizard
just sequences them.

First run, on full consent, reaches the same end state as:

```shell
nabu init
nabu install all
nabu backfill --tool all
nabu index --once          # imported events are not searchable until indexed
nabu mcp install all
```

After importing history the wizard builds the lexical index automatically, so
search works immediately. Semantic embedding stays opt-in (it is the slow path);
run a later `nabu index --once` to embed once the model is installed.

Properties:

- **Consent-gated.** Nothing is created, installed, or changed without an
  explicit confirm at that step. Declining a step leaves it unchanged and
  continues.
- **Idempotent.** Re-running on a configured machine reports current state and
  offers repair/uninstall via *Manage integrations* — it never duplicates an
  install. User settings in `config.toml` are preserved.
- **TTY-only.** In a pipe, hook, or CI context the wizard refuses with a
  non-zero exit and prints the explicit commands above; it never prompt-blocks
  automation.
- **Redaction stays opt-in.** *Settings* is a read-only inspector that reports
  effective settings and how to change each; it never enables redaction.
- **One screen at a time.** The wizard redraws a single frame per screen — a
  constant header (brand, live capture status, store path) above the menu or the
  current action — rather than accumulating a scrollback log. Each menu choice
  opens its own screen and returns to the same menu on `↵`.

*Get started* runs four consented steps — *Storage*, *Capture* (per-tool hooks),
*Backfill*, *Connect* (MCP) — and closes with a summary of what is configured and
the next command to run. Each step states the file it will change and routes the
write through the install/backfill/MCP functions; the full diff stays one explicit
command away (`nabu install <tool> --dry-run`).

Top menu: *Get started*, *Manage integrations*, *Backfill history*, *Settings*,
*Health check*, *Connect agents (MCP)*, *Quit* — each maps to the corresponding
explicit command.

### `nabu ingest` — write events into the store

Two subcommands: `hook` (read one event payload from stdin) and `file` (read a
file of native records). This is the low-level path that capture adapters and
backfill use internally; you rarely call it by hand.

#### `nabu ingest hook` — ingest one live hook payload from stdin

```shell
nabu ingest hook --tool codex|claude|opencode
```

| Flag | Description | Default |
| --- | --- | --- |
| `--tool <TOOL>` | Source tool: `codex`, `claude`, or `opencode`. Required. | — |

```shell
echo "$HOOK_JSON" | nabu ingest hook --tool claude
```

#### `nabu ingest file` — ingest a file of native records

```shell
nabu ingest file --tool codex|claude|opencode --source <SOURCE> --path PATH
```

| Flag | Description | Default |
| --- | --- | --- |
| `--tool <TOOL>` | Source tool: `codex`, `claude`, or `opencode`. Required. | — |
| `--source <SOURCE>` | Record format: `backfill`, `exec_json`, `app_server`, `event_stream`, `transcript_tail`. Required. | — |
| `--path <PATH>` | File to read. Required. | — |

Codex exact streaming ingestion:

```shell
codex exec --json "..." > codex-run.jsonl
nabu ingest file --tool codex --source exec_json --path codex-run.jsonl
nabu ingest file --tool codex --source app_server --path app-server-notifications.jsonl
```

`exec_json` and `app_server` preserve native stream records as raw payloads. `item/agentMessage/delta` records are stored as `assistant.delta` in source order; `turn.completed` usage objects remain available in raw export and search.

### `nabu index` — build the derived index from raw files

```shell
nabu index --once [--no-embed] [--json-progress]
nabu index --watch [--no-embed] [--json-progress]
```

| Flag | Description | Default |
| --- | --- | --- |
| `--once` | Run a single index pass over changed raw files, then exit. | — |
| `--watch` | Watch raw files and index continuously. | — |
| `--no-embed` | Build lexical FTS/index state only; write no vector embeddings. | off |
| `--json-progress` | Emit newline-delimited JSON progress events on stderr instead of human text. | off |

```shell
nabu index --once
nabu index --once --no-embed
nabu index --watch --json-progress
```

Indexing records derived checkpoints for canonical raw JSONL files. A repeated
`index --once` skips unchanged raw files by source identity, size, and mtime;
changed files are re-scanned from the top and remain idempotent through
`dedupe_key`.

Long-running index commands write progress to stderr. When semantic mode is built
and the local model is present, the embedding pass first emits an
`embed.index` / `embedding_plan` event with the unembedded-unit count and
one-time local CPU cost. It then emits `loading_model` and `embedding` phases,
including embedded/total units, units/sec, ETA, intra-op thread count, batch
size, and write chunk size. Add `--json-progress` for newline-delimited JSON
progress events suitable for agents and scripts.

Use `--no-embed` to build lexical FTS/index state only. In a semantic build this
still materializes derived vector units, but writes zero vector embeddings and
leaves semantic search unavailable until a later default index pass embeds the
pending units.

### `nabu backfill` — import native history the upstream agent still holds

```shell
nabu backfill --tool codex|claude|opencode|all [--since DATE_OR_DURATION] [--path PATH] [--dry-run] [--json-progress]
```

| Flag | Description | Default |
| --- | --- | --- |
| `--tool <TOOL>` | `codex`, `claude`, `opencode`, or `all`. Required. | — |
| `--since <SINCE>` | Only import sessions at or after this duration/date/timestamp. | all |
| `--path <PATH>` | Read from this path instead of the native local roots. | native roots |
| `--dry-run` | Print per-session coverage; append no raw events and write no checkpoints. | off |
| `--json-progress` | Switch stderr progress to newline-delimited JSON. | off |

```shell
nabu backfill --tool all
nabu backfill --tool claude --since 30d
nabu backfill --tool codex --dry-run
```

Without `--path`, backfill scans native local roots: `$CODEX_HOME/sessions`, `$CODEX_HOME/archived_sessions`, `$CLAUDE_CONFIG_DIR/projects` or `~/.claude/projects`, and `~/.local/share/opencode/`.

OpenCode server reconciliation is disabled unless `NABU_OPENCODE_URL` or `[opencode] server_url = "..."` in `config.toml` is set. When configured, backfill discovers local OpenCode session ids and fetches `GET /session/:id/message`. Fetched server messages are appended directly to canonical raw session files; nabu does not keep a second copy of the API response in spool storage.

If the configured OpenCode server is unreachable or returns a non-2xx response, nabu writes a warning to stderr, skips that session's server reconciliation, and continues. With no URL configured, no OpenCode network request is made.

Use `--dry-run` to print per-session `on_disk`, `captured`, `missing`, `partial`, and `would_import` coverage without appending raw events or writing checkpoint rows.

Backfill and backfill dry-run stream progress to stderr and reserve stdout for the final report, so agent callers can parse stdout without losing progress visibility. `--json-progress` switches stderr progress to newline-delimited JSON.

### `nabu search` — query indexed history

```shell
nabu search QUERY [OPTIONS]
```

| Flag | Description | Default |
| --- | --- | --- |
| `<QUERY>` | Search text. Required positional argument. | — |
| `--tool <TOOL>` | Restrict to `codex`, `claude`, or `opencode`. | all |
| `--session <SESSION>` | Restrict to a session id. | all |
| `--cwd <CWD>` | Restrict to a working directory. | all |
| `--since <SINCE>` | Restrict to events at or after this duration/date/timestamp. | all |
| `--type <CANONICAL_TYPE>` | Restrict to a canonical event type. | all |
| `--file <FILE>` | Restrict to events mentioning a file path. | all |
| `--command <COMMAND>` | Restrict to events mentioning a command. | all |
| `--mode <MODE>` | `auto`, `lexical`, or `hybrid`. | `auto` |
| `--corroborate` | Annotate hits with read-only local git existence checks. | off |
| `--limit <LIMIT>` | Maximum hits to return. | `10` |
| `--offset <OFFSET>` | Skip this many hits (paging). | `0` |
| `--full` | Restore full event payloads (otherwise `payload` is `null`). | off |
| `--include-deltas` | Include `assistant.delta` events. | off |
| `--no-dedupe` | Do not collapse duplicate hits. | off |
| `--max-snippet-chars <N>` | Cap snippet length. | `240` |
| `--format <FORMAT>` | `human`, `json`, or `markdown`. | `human` |

```shell
nabu search "retry backoff"
nabu search "auth bug" --tool claude --since 7d --limit 20
nabu search "deploy script" --format json --full
```

Search is citation-first by default: JSON keeps the stable `payload` key but sets it to `null`. Use `--full` to restore payloads, `--offset` to page, and `show --around-line` or `get_event` to drill into a hit.

`--mode auto` is the default. In the default build, `auto` applies lexical BM25 because the semantic backend is not compiled. `--mode lexical` always uses BM25 and never loads a model. `--mode hybrid` requires a compatible semantic build, local model cache, and vector index; otherwise it fails with `SEMANTIC_UNAVAILABLE` instead of silently falling back.

Use `--corroborate` to annotate results with local git existence checks for mentioned commits, branches, and files. The check is read-only, never fetches, never calls a forge API, and never changes ranking or filtering. PR references such as `#123` or `/pull/123` are reported as `unresolved` with `reason=needs_network` because resolving them requires network access.

### `nabu embed` — manage the local semantic model

Three subcommands: `status`, `download`, `prune`.

#### `nabu embed status` — report model presence and footprint

```shell
nabu embed status
```

No flags beyond `--home`.

#### `nabu embed download` — fetch the embedding model

```shell
nabu embed download [--model MODEL] --yes [--json-progress]
```

| Flag | Description | Default |
| --- | --- | --- |
| `--model <MODEL>` | Model id to fetch. | `embeddinggemma-300m-q4` |
| `--yes` | Confirm after the disclosure. Required to proceed. | off |
| `--json-progress` | Emit newline-delimited JSON progress on stderr. | off |

#### `nabu embed prune` — delete the downloaded model

```shell
nabu embed prune --yes [--json-progress]
```

| Flag | Description | Default |
| --- | --- | --- |
| `--yes` | Confirm deletion. Required to proceed. | off |
| `--json-progress` | Emit newline-delimited JSON progress on stderr. | off |

```shell
nabu embed status
nabu embed download --yes
nabu embed prune --yes
```

The model is never downloaded by search, index, capture, doctor, export, or MCP reads. Acquisition is explicit and stores model files under `models/`. Before fetching, `embed download` prints the model id, source repository, Gemma Terms summary, and measured current local footprint; `--yes` is required after that disclosure. The M5 model is Google EmbeddingGemma 300M Q4 via fastembed/ONNX, truncated to 256 dimensions; note the model license separately from crate licenses and expect roughly a few hundred MB of local model cache. Index-time embedding configures ONNX Runtime intra-op threads from the detected physical core count, capped by available parallelism; set `NABU_SEMANTIC_INTRA_THREADS=N` only when you need to cap or benchmark CPU use.

Semantic indexing stores compact derived vector-unit text rows keyed by content
hash so embedding can resume without rehydrating each unit from raw JSONL. These
rows live inside the derived SQLite index, are safe to delete, and are rebuilt
from canonical raw files.

### `nabu show` — print a normalized session view

```shell
nabu show TOOL SESSION_ID [OPTIONS]
```

| Flag | Description | Default |
| --- | --- | --- |
| `<TOOL>` | `codex`, `claude`, or `opencode`. Required positional. | — |
| `<SESSION_ID>` | Session id. Required positional. | — |
| `--limit-events <N>` | Maximum events to print. | `100` |
| `--after-raw-line <N>` | Start after this raw line number. | start |
| `--around-line <N>` | Center the window on this raw line. | — |
| `--before <N>` | Lines of context before `--around-line`. | `5` |
| `--after <N>` | Lines of context after `--around-line`. | `5` |
| `--type <CANONICAL_TYPE>` | Restrict to a canonical event type. | all |
| `--include-deltas` | Include `assistant.delta` events. | off |
| `--corroborate` | Annotate with read-only local git existence checks. | off |
| `--format <FORMAT>` | `human`, `json`, or `markdown`. | `human` |

```shell
nabu show claude 019a4b44-cc3b-7c51-8944-a7d7ebb9e6fe
nabu show codex SESSION_ID --around-line 420 --before 10 --after 10
```

Session views hide `assistant.delta` by default. Use `--include-deltas` for the full normalized stream; `export` always preserves full raw fidelity.

### `nabu tail` — read a session's raw JSONL

```shell
nabu tail TOOL SESSION_ID [--follow]
```

| Flag | Description | Default |
| --- | --- | --- |
| `<TOOL>` | `codex`, `claude`, or `opencode`. Required positional. | — |
| `<SESSION_ID>` | Session id. Required positional. | — |
| `--follow` | Stream new raw events as they are appended. | off |

```shell
nabu tail claude SESSION_ID
nabu tail codex SESSION_ID --follow
```

### `nabu export` — emit a session at full fidelity

```shell
nabu export TOOL SESSION_ID --format jsonl|markdown [--redact]
```

| Flag | Description | Default |
| --- | --- | --- |
| `<TOOL>` | `codex`, `claude`, or `opencode`. Required positional. | — |
| `<SESSION_ID>` | Session id. Required positional. | — |
| `--format <FORMAT>` | `jsonl` or `markdown`. Required. | — |
| `--redact` | Apply redaction for shareable output. | off |

```shell
nabu export claude SESSION_ID --format markdown
nabu export codex SESSION_ID --format jsonl --redact > session.jsonl
```

### `nabu doctor` — report capture and store health

```shell
nabu doctor [--tool codex|claude|opencode|all] [--deep] [--json]
```

| Flag | Description | Default |
| --- | --- | --- |
| `--tool <TOOL>` | `codex`, `claude`, `opencode`, or `all`. | `all` |
| `--deep` | Run full SQLite integrity and counts (slower). | off |
| `--json` | Emit machine-readable JSON. | off |

```shell
nabu doctor
nabu doctor --tool claude --deep
nabu doctor --json
```

`doctor --json` includes `storage_footprint` so users can see local raw, index, vector, spool, blob, model, canonical, derived, and total byte usage.

Doctor is fast by default and reports `integrity=quick`; use `--deep` for full SQLite integrity and counts.

### `nabu install` / `nabu uninstall` — manage live capture hooks

```shell
nabu install codex|claude|opencode|all [--dry-run]
nabu uninstall codex|claude|opencode|all [--dry-run]
```

| Flag | Description | Default |
| --- | --- | --- |
| `<TOOL>` | `codex`, `claude`, `opencode`, or `all`. Required positional. | — |
| `--dry-run` | Print the diff that would be applied; write nothing. | off |

```shell
nabu install all --dry-run
nabu install claude
nabu uninstall codex
```

### `nabu purge` — delete history by session, date, or everything

```shell
nabu purge --session TOOL:SESSION_ID
nabu purge --before DATE_OR_DURATION
nabu purge --all [--keep-model] [--keep-config] [--dry-run] [--yes]
```

Exactly one of `--session`, `--before`, or `--all` is required.

| Flag | Description | Default |
| --- | --- | --- |
| `--session <TOOL:SESSION_ID>` | Delete one session, e.g. `claude:SESSION_ID`. | — |
| `--before <DATE_OR_DURATION>` | Delete sessions older than this duration/date/timestamp. | — |
| `--all` | Uninstall all hooks and delete the store. Previews and asks for confirmation first. | — |
| `--keep-model` | With `--all`: keep the downloaded embedding model under `models/`. | off |
| `--keep-config` | With `--all`: keep `config.toml` (your settings). | off |
| `--dry-run` | With `--all`: print the full preview and exit without deleting. | off |
| `--yes` | With `--all`: skip the typed confirmation (required in non-interactive use). | off |

```shell
nabu purge --session claude:019a4b44-cc3b-7c51-8944-a7d7ebb9e6fe
nabu purge --before 90d
nabu purge --all --dry-run
nabu purge --all --keep-model --yes
```

`purge --all` is the inverse of install + init. It uninstalls the hooks from each tool's own config (via the contracted uninstall path) and deletes the store under the home. Behavior:

- Prints a full preview first — hooks to remove, every store artifact with its size, and which entries are kept — then requires a typed `purge` confirmation. `--dry-run` prints the preview and exits without deleting; `--yes` skips the prompt (required in non-interactive contexts, where it otherwise refuses and exits non-zero).
- Removes only the closed set of nabu artifacts (`raw`, `index`, `spool`, `checkpoints`, `blobs`, `logs`, `backups`, `models`, `config.toml`). The home directory itself and any non-nabu files inside it are left in place and reported as untouched.
- `--keep-model` preserves the downloaded embedding model under `models/`; `--keep-config` preserves `config.toml` (your settings).
- `raw/` is flagged irreversible: it is the authoritative capture, and any session the native tool store no longer holds cannot be recovered after removal. All other artifacts are derived and rebuildable from `raw/`.
- The installed `nabu` binary is not removed; it lives outside the store.
- Refuses to run against the filesystem root, `$HOME`, or any directory that carries no nabu marker (`config.toml`/`index`/`raw`), so a mistyped `--home` errors instead of deleting.

### `nabu bench` — micro-benchmark ingest and search

Two subcommands: `ingest` and `search`.

#### `nabu bench ingest`

```shell
nabu bench ingest --events PATH [--seed-events N] [--iterations N] [--json-progress]
```

| Flag | Description | Default |
| --- | --- | --- |
| `--events <EVENTS>` | Event fixture file to ingest. Required. | — |
| `--seed-events <N>` | Pre-seed the store with this many events before timing. | `0` |
| `--iterations <N>` | Timed iterations. | `1000` |
| `--json-progress` | Emit newline-delimited JSON progress on stderr. | off |

#### `nabu bench search`

```shell
nabu bench search --query TEXT [--iterations N] [--limit N] [--json-progress]
```

| Flag | Description | Default |
| --- | --- | --- |
| `--query <QUERY>` | Search text to benchmark. Required. | — |
| `--iterations <N>` | Timed iterations. | `100` |
| `--limit <N>` | Hits per query. | `10` |
| `--json-progress` | Emit newline-delimited JSON progress on stderr. | off |

```shell
nabu bench ingest --events fixtures/bench/events.jsonl
nabu bench ingest --events fixtures/bench/events.jsonl --seed-events 10000 --iterations 1000 --json-progress
nabu bench search --query "fixture marker" --home fixtures/acceptance-home --iterations 100 --limit 10
```

## Gotchas

### Backfill only recovers history the upstream agent still keeps

`backfill` reads each tool's native session store. It cannot recover sessions the upstream agent has already deleted. Capture must be installed before the upstream retention window expires; a session that predates capture *and* its agent's retention window is unrecoverable from both the native store and nabu.

Claude Code prunes its own transcripts under `~/.claude/projects/` on startup using `cleanupPeriodDays` (default 30 days; unset means 30). On 2026-06-18 a store with `cleanupPeriodDays` unset retained no transcript older than 2026-05-19 — a rolling 30-day window. A session older than that window that was never captured live leaves nothing on disk for `backfill` to import. To retain native history beyond the capture-install date, raise `cleanupPeriodDays` in `~/.claude/settings.json`.

Codex (`$CODEX_HOME/sessions`, `archived_sessions`) and OpenCode apply their own retention. The rule is the same: install capture early rather than relying on `backfill` to reach back past an agent's cleanup window.

## MCP

The `nabu mcp` command group has four subcommands: `serve`, `install`,
`uninstall`, `validate`.

### `nabu mcp serve` — run the MCP server

MVP MCP transport is stdio only.

```shell
nabu mcp serve --transport stdio
```

| Flag | Description | Default |
| --- | --- | --- |
| `--transport <TRANSPORT>` | Transport to serve. Only `stdio` is supported. Required. | — |

MCP read tools `search_history`, `get_session`, `get_event`, and
`recall_answer` accept `corroborate=true` for the same local read-only git
annotations as CLI `--corroborate`. This is not a separate MCP tool and does not
fetch PR or forge state.

### `nabu mcp install` / `nabu mcp uninstall` — manage MCP client entries

```shell
nabu mcp install codex|claude|opencode|all [--dry-run]
nabu mcp uninstall codex|claude|opencode|all [--dry-run]
```

| Flag | Description | Default |
| --- | --- | --- |
| `<TOOL>` | `codex`, `claude`, `opencode`, or `all`. Required positional. | — |
| `--dry-run` | Print the diff that would be applied; write nothing. | off |

```shell
nabu mcp install all --dry-run
nabu mcp install claude
nabu mcp uninstall codex
```

### `nabu mcp validate` — probe MCP client wiring

```shell
nabu mcp validate codex|claude|opencode|all [--json]
```

| Flag | Description | Default |
| --- | --- | --- |
| `<TOOL>` | `codex`, `claude`, `opencode`, or `all`. Required positional. | — |
| `--json` | Emit machine-readable JSON. | off |

```shell
nabu mcp validate all --json
```

`mcp validate` is read-only. It reports upstream client probe evidence, whether the `nabu` MCP entry is installed, and an in-process fixture MCP handshake/query result.

MVP MCP tools:

- `search_history`
- `list_sessions`
- `get_session`
- `export_session`
- `get_event`
- `history_doctor`
- `recall_answer`

`search_history` is citation-first and payload-light by default. MCP callers that need prior payload behavior must pass `include_payload=true`; use `offset` for paging and `get_session` with `around_raw_line` for context windows. `mode` accepts `auto`, `lexical`, or `hybrid` with the same fallback/error behavior as CLI search.

`recall_answer` runs search, pulls bounded `get_session` context windows around top hits, dedupes overlapping context, and returns cited material. It does not generate prose, call an LLM, mutate history, or make network requests.

MVP MCP resources:

- `nabu://sessions`
- `nabu://sessions/{tool}/{session_id}`
- `nabu://schema/tools`

MVP MCP prompts:

- `recall_project_history`
- `prepare_handoff_summary`

## Privacy

nabu is local-first. MVP commands do not send transcript data over the network by default, and there is no product telemetry or central database. Raw files are full fidelity and may contain secrets. Use `export --redact` for agent-facing or shareable output, and `purge` to delete by session or date.
