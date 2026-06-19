# nabu-cli &nbsp;𒀭𒀝

**The scribe for your coding agents.** `nabu-cli` installs the `nabu` binary: a
local-first, device-only, cross-agent history keeper for Codex, Claude Code, and
OpenCode. Append-only per-session JSONL is authoritative; a rebuildable SQLite FTS5
index serves search to you on the CLI and to agents over MCP — with a citation
(session, tool, raw line) on every result.

*Nabû* is the Babylonian god of scribes and writing, written 𒀭𒀝 (dAG) in cuneiform.

## Install

```shell
cargo install nabu-cli      # installs the `nabu` binary
```

## Quickstart

```shell
nabu init                                  # create the local store (~/.nabu)
nabu wizard                                # guided install of capture hooks + MCP
nabu search "auth bug"                     # find prior work, with citations
nabu mcp serve --transport stdio           # expose history to agents
```

Optional semantic/hybrid search is behind a default-off `semantic` cargo feature.

## More

- Full command reference: [docs/USAGE.md](https://github.com/suleymanozkeskin/nabu/blob/main/docs/USAGE.md)
- Project README: <https://github.com/suleymanozkeskin/nabu>

## License

Licensed under either of MIT or Apache-2.0 at your option.
