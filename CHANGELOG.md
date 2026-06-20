# Changelog

All notable user-facing changes should be recorded here. Release narratives live
in `docs/release-notes.md`.

## Unreleased

- Add CI supply-chain checks with `cargo-deny`.
- Make semantic retrieval-quality acceptance explicit and runnable from a
  separate scheduled/manual workflow.
- Fix MCP request failure handling and FTS rebuild recovery.
- Preserve cited prefixes for oversized MCP `get_session`, `get_event`, and
  markdown `export_session` responses.
- Validate OpenCode sync URLs before writing config and preserve JSONC comments,
  same-line comments, BOMs, and CRLF line endings when installing OpenCode MCP
  config.
- Cache redaction regexes and reduce whole-file reads in purge/Codex JSON
  ingest paths.
- Add dedupe property coverage and project security/contributing docs.

## 0.1.0

- Initial public release. See `docs/release-notes.md`.
