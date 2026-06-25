# Contributing

Thanks for working on nabu.

## Development Checks

Run the default checks before sending changes:

```shell
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo test --workspace --features semantic
```

Supply-chain checks use `cargo-deny`:

```shell
cargo deny --all-features check all
```

### Git hooks (optional)

Tracked hooks under `.githooks/` run the same `fmt`/`clippy` gates as CI so
style and lint failures surface locally. They are not enabled automatically;
opt in once per clone:

```shell
sh scripts/setup-hooks.sh
```

This sets `core.hooksPath` to `.githooks/`. The `pre-commit` hook runs
`cargo fmt --all --check` (fast); the `pre-push` hook runs
`cargo clippy --workspace --all-targets -- -D warnings`. Bypass a hook with
`git commit --no-verify` / `git push --no-verify` when needed.

The model-backed semantic acceptance tests are ignored during normal test runs.
Run them only when a local `embeddinggemma-300m-q4` cache is available:

```shell
NABU_SEMANTIC_TEST_HOME=/path/to/nabu-home \
  cargo test -p nabu-core --features semantic semantic_acceptance -- --ignored --nocapture
```

## Expectations

- Preserve raw capture fidelity. Normalize for indexing, but keep upstream
  payloads recoverable.
- Keep user config edits narrow and idempotent. Back up files before mutating
  agent-owned config.
- Treat raw history, blobs, exports, and backups as sensitive.
- Add focused tests for parser, dedupe, purge, MCP, and config-write changes.
- Avoid broad refactors when fixing narrow behavior.

## Release Notes

User-facing changes should update `docs/release-notes.md` or `CHANGELOG.md` as
appropriate.
