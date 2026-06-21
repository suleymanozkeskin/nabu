# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0](https://github.com/suleymanozkeskin/nabu/releases/tag/nabu-mcp-v0.1.0) - 2026-06-21

### Fixed

- make fast doctor O(1) in index size
- OpenCode capture + concurrent MCP request handling

### Other

- Merge branch 'feat/native-handoff-command' into integration/issue-1-all
- Merge branch 'feat/recall-answer-clarify' into integration/issue-1-all
- Merge branch 'feat/get-session-cursor' into integration/issue-1-all
- Merge branch 'feat/semantic-search-mode' into integration/issue-1-all
- Merge branch 'feat/search-tool-all' into integration/issue-1-all
- Merge branch 'feat/provenance-ref-indexing' into integration/issue-1-all
- Merge branch 'feat/richer-list-sessions-metadata' into integration/issue-1-all
- Merge branch 'feat/surface-session-summaries' into integration/issue-1-all
- Merge branch 'feat/configurable-snippet-triage' into integration/issue-1-all
- *(workspace)* mark internal crates doc(hidden); trim nabu-core surface
- Harden CLI and MCP edge cases
- Harden MCP responses and config editing
- Fix MCP request failures and FTS recovery
- Guard deep MCP doctor on large indexes
- optimize history flows across ingest, search, backfill, and MCP
- Initial public release
