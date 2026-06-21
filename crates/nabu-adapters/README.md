# nabu-adapters &nbsp;𒀭𒀝

Capture adapters for [nabu](https://github.com/suleymanozkeskin/nabu), a
local-first, cross-agent history keeper for coding agents.

This crate installs and removes capture integrations for Codex, Claude Code, and
OpenCode: hook entries and the OpenCode plugin. Every config mutation is backed up
first, supports a dry-run diff, and touches only nabu-owned entries — never other
user settings.

For the CLI and end-user docs, install [`nabu-cli`](https://crates.io/crates/nabu-cli)
or see <https://github.com/suleymanozkeskin/nabu>.

## License

Licensed under either of MIT or Apache-2.0 at your option.
