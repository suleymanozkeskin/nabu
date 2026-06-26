# Changelog

All notable user-facing changes should be recorded here. Release narratives live
in `docs/release-notes.md`.

## Unreleased

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
