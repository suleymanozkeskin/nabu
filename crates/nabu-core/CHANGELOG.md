# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0](https://github.com/suleymanozkeskin/nabu/releases/tag/nabu-core-v0.1.0) - 2026-06-21

### Added

- *(wizard)* multi-select, back nav, live health, actionable settings

### Fixed

- *(tests)* repair union-merge interleaving of summary/list_sessions tests
- *(core)* session search filter fails open on prefix and filename
- *(core)* OR-semantics search so specificity never collapses recall
- make fast doctor O(1) in index size
- OpenCode capture + concurrent MCP request handling
- *(backfill)* skip source files that vanish mid-scan instead of aborting

### Other

- Merge branch 'feat/native-handoff-command' into integration/issue-1-all
- Merge branch 'feat/get-session-cursor' into integration/issue-1-all
- Merge branch 'feat/semantic-search-mode' into integration/issue-1-all
- Merge branch 'feat/provenance-ref-indexing' into integration/issue-1-all
- Merge branch 'feat/richer-list-sessions-metadata' into integration/issue-1-all
- Merge branch 'feat/surface-session-summaries' into integration/issue-1-all
- Merge branch 'feat/configurable-snippet-triage' into integration/issue-1-all
- Merge branch 'fix/strip-tool-call-noise' into integration/issue-1-all
- Merge branch 'fix/dedup-adjacent-twins' into integration/issue-1-all
- *(workspace)* mark internal crates doc(hidden); trim nabu-core surface
- *(core)* correct session-prefix tier comment (returns all matches)
- *(nabu-core)* import HashSet in semantic for the Linux core-count path
- *(nabu-core)* relocate residual DTOs, readers, and helpers out of lib.rs
- *(nabu-core)* sweep mod tests into tests.rs
- *(nabu-core)* extract semantic implementation into semantic module
- *(nabu-core)* extract backfill into directory module with per-format submodules
- *(nabu-core)* extract history search into search directory module
- *(nabu-core)* extract health checks into doctor module
- *(nabu-core)* extract store purge into purge module
- *(nabu-core)* extract indexing pipeline into index module
- *(nabu-core)* extract ingest pipeline into ingest module
- *(nabu-core)* extract read and export into modules
- *(nabu-core)* extract redaction into redact module
- *(nabu-core)* extract OpenCode config I/O into config module
- *(nabu-core)* extract canonicalization/document extraction into document module
- *(nabu-core)* extract database lifecycle into db module
- *(nabu-core)* extract semantic seam into semantic_api module
- *(nabu-core)* extract event identity into identity module
- *(nabu-core)* extract path/permission helpers into paths module
- *(nabu-core)* extract JSON accessors into json module
- *(nabu-core)* extract DTO/option/report structs into options module
- *(nabu-core)* extract event model into event module
- *(nabu-core)* extract error type into error module
- *(nabu-core)* capture public-API snapshot and wire drift guard
- Harden CLI and MCP edge cases
- Harden MCP responses and config editing
- Add semantic acceptance and supply-chain gates
- Fix MCP request failures and FTS recovery
- optimize history flows across ingest, search, backfill, and MCP
- Initial public release
