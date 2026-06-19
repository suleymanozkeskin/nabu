# nabu-core &nbsp;𒀭𒀝

Core storage, indexing, and search for [nabu](https://github.com/suleymanozkeskin/nabu),
a local-first, cross-agent history keeper for coding agents.

This crate owns the data model: append-only per-session JSONL is the source of
truth, and a rebuildable SQLite FTS5 index is derived from it. It provides home
resolution, ingest, event normalization, the index schema, and citation-first
search. Optional semantic/hybrid search is behind a default-off `semantic` feature.

For the CLI and end-user docs, install [`nabu-cli`](https://crates.io/crates/nabu-cli)
or see <https://github.com/suleymanozkeskin/nabu>.

## License

Licensed under either of MIT or Apache-2.0 at your option.
