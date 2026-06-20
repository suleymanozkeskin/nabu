# nabu &nbsp;𒀭𒀝

[![CI](https://github.com/suleymanozkeskin/nabu/actions/workflows/ci.yml/badge.svg)](https://github.com/suleymanozkeskin/nabu/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/nabu-cli.svg)](https://crates.io/crates/nabu-cli)
[![docs.rs](https://img.shields.io/docsrs/nabu-core)](https://docs.rs/nabu-core)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
![MSRV](https://img.shields.io/badge/MSRV-1.88-blue.svg)

**The scribe for your coding agents.** Durable, local, cross-tool history of
everything Codex, Claude Code, and OpenCode do — searchable long after the
context window, the process, or the tool is gone.

*Nabû* is the Babylonian god of scribes and writing, written **𒀭𒀝 (dAG)** in
cuneiform. It does one job well: inscribe what happened so it outlasts memory.

![nabu demo — title, the problem, and a real agent recovering a past fix over MCP](demo/demo-full.gif)

## What it is

Coding agents forget. Compaction drops detail, sessions end, and switching tools
loses context. nabu captures every session as append-only JSONL, indexes it
into a rebuildable SQLite full-text index, and serves it back — to you on the CLI,
and to other agents over MCP — with a citation (session, tool, raw line) on every
result.

- **Local-first.** No telemetry, no cloud, no central database. Your history stays
  on your machine.
- **Full fidelity of what each tool exposes.** Raw JSONL is the source of truth;
  the index is derived and rebuildable. nabu preserves upstream payloads before
  best-effort normalization.
- **Cross-agent.** One normalized event model over Codex, Claude Code, and
  OpenCode.
- **Agent-first.** A read-only MCP server (`search_history`, `get_session`, …) so a
  new session can recover what an old one learned.

## Install

```shell
cargo install nabu-cli      # installs the `nabu` binary
```

Optional semantic/hybrid search is behind a default-off `semantic` cargo feature.

## Quickstart

```shell
nabu init                 # create the local store (~/.nabu)
nabu wizard               # guided install of capture hooks + MCP (or: nabu install all)
nabu search "auth bug"    # find prior work, with citations
nabu mcp serve --transport stdio   # expose history to agents
```

## How to use

Full command reference and the MCP surface: **[docs/USAGE.md](docs/USAGE.md)**.

## Privacy

Raw history is full fidelity and may contain secrets. It never leaves the machine
by default. Use `export --redact` for anything shared, and `purge` to delete by
session or date.

## Status

MVP, in active development. Capture, indexing, citation-first search, and the MCP
server work today.

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your
option.
