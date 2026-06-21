# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0](https://github.com/suleymanozkeskin/nabu/releases/tag/nabu-cli-v0.1.0) - 2026-06-21

### Added

- surface provenance refs via ref= search filter (CLI + MCP)
- *(wizard)* multi-select, back nav, live health, actionable settings
- *(wizard)* auto-index after backfill so imports are searchable

### Fixed

- *(core)* OR-semantics search so specificity never collapses recall
- make fast doctor O(1) in index size
- *(mcp)* idempotent install that clears legacy tupsharrum entries

### Other

- Merge branch 'feat/native-handoff-command' into integration/issue-1-all
- Merge branch 'feat/semantic-search-mode' into integration/issue-1-all
- Merge branch 'feat/provenance-ref-indexing' into integration/issue-1-all
- Merge pull request #4 from suleymanozkeskin/fix/mcp-discovery-recall
- *(cli)* make env-test lock poison-tolerant
- *(cli)* slim run dispatch by extracting large command arms
- *(cli)* collapse the config-change print triple into one helper
- *(cli)* extract output rendering into render module
- *(cli)* extract backfill subsystem into backfill module
- *(cli)* extract MCP config subsystem into mcp_config module
- *(cli)* replace per-tool path resolvers with ToolLayout trait
- *(cli)* extract benchmark commands into bench module
- *(cli)* extract progress rendering into progress module
- *(cli)* extract config backup/write helpers into backup module
- *(cli)* extract OpenCode HTTP client into opencode_http module
- *(cli)* extract JSONC editor into jsonc_edit module
- *(cli)* extract test-support helpers into shared module
- Harden CLI and MCP edge cases
- Harden MCP responses and config editing
- optimize history flows across ingest, search, backfill, and MCP
- Initial public release
