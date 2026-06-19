# nabu-mcp &nbsp;𒀭𒀝

Read-only MCP server for [nabu](https://github.com/suleymanozkeskin/nabu), a
local-first, cross-agent history keeper for coding agents.

This crate serves a coding agent's own history back to a new session over the
Model Context Protocol (stdio): `search_history`, `get_session`, `recall_answer`,
`list_sessions`, `get_event`, `export_session`, and `history_doctor`. Every result
carries a citation (session, tool, raw line). The server is read-only — it never
mutates the store.

For the CLI and end-user docs, install [`nabu-cli`](https://crates.io/crates/nabu-cli)
or see <https://github.com/suleymanozkeskin/nabu>.

## License

Licensed under either of MIT or Apache-2.0 at your option.
