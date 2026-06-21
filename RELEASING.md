# Releasing nabu

nabu publishes four crates to crates.io so that `cargo install nabu-cli`
resolves its dependencies. They publish in dependency order:

```
nabu-core → nabu-adapters → nabu-mcp → nabu-cli
```

All four share one version via `[workspace.package].version` and release
together. The internal crates (`nabu-core`, `nabu-adapters`, `nabu-mcp`) are
`doc(hidden)` and exist only so the `nabu` binary resolves its deps; they are
still published. A default `cargo install nabu-cli` builds without the
`semantic` feature, so it does not pull the ONNX/embedding stack.

## One-time setup

1. Create a crates.io API token at <https://crates.io/settings/tokens>. Scope the
   CI token to publish only.
2. **Local (first release):** run `cargo login` and paste the token. It is stored
   in `~/.cargo/credentials.toml`, outside the repo.
3. **CI (ongoing):** add the token as a repository secret named
   `CARGO_REGISTRY_TOKEN` (Settings → Secrets and variables → Actions, or
   `gh secret set CARGO_REGISTRY_TOKEN`). **Never commit the token** — not in
   `Cargo.toml`, the workflow, or a tracked file.

## First release (manual, supervised)

The first publish claims the crate names and is irreversible (you can only
`yank`, never delete), so do it by hand in dependency order, with dry-runs first.
From a clean checkout of `main`:

```shell
# nabu-core has no internal deps, so its dry-run verifies fully:
cargo publish -p nabu-core --dry-run

# Downstream dry-runs cannot fully verify before nabu-core is on crates.io —
# their verify build resolves nabu-core from the registry, where it does not
# exist yet ("no matching package named `nabu-core`"). Check packaging only:
cargo publish -p nabu-adapters --dry-run --no-verify
cargo publish -p nabu-mcp      --dry-run --no-verify
cargo publish -p nabu-cli      --dry-run --no-verify

# then publish in order; each downstream publish verifies against the
# now-published upstream. Wait ~30s between each so the index propagates:
cargo publish -p nabu-core
cargo publish -p nabu-adapters
cargo publish -p nabu-mcp
cargo publish -p nabu-cli
```

(The `release-plz` automation below handles this ordering and index propagation
for you; the manual dance only matters for the very first release.)

Verify in a clean environment: `cargo install nabu-cli` installs the `nabu`
binary.

## Ongoing releases (automated, release-plz)

`.github/workflows/release-plz.yml` runs on every push to `main`:

1. On normal merges, release-plz opens or updates a **release PR** that bumps the
   workspace version and updates `CHANGELOG.md` from the Conventional Commits
   since the last release.
2. Review and merge that PR. The version-bump commit lands on `main`, and the
   `release` job publishes every changed crate to crates.io in dependency order
   and creates the GitHub release and tags.

After the first release you never run `cargo publish` by hand — you just merge
the release PR. Configuration lives in `release-plz.toml`.

### Token note for CI

The `release` job uses `CARGO_REGISTRY_TOKEN` (repo secret) to publish. The
release PR is opened with the built-in `GITHUB_TOKEN`. Note: PRs opened by
`GITHUB_TOKEN` do **not** trigger other workflows, so CI will not run on the
release PR itself. If you want CI to run on release PRs, create a fine-grained
PAT with `contents: write` + `pull-requests: write`, store it as
`RELEASE_PLZ_TOKEN`, and pass it as `GITHUB_TOKEN:` to the release-plz steps
instead of `secrets.GITHUB_TOKEN`.

## Optional: prebuilt binaries (cargo-dist)

crates.io requires users to have a Rust toolchain. To also ship prebuilt
binaries on GitHub Releases (macOS/Linux/Windows, no local compile of the
semantic/ONNX stack), add [cargo-dist](https://opensource.axo.dev/cargo-dist/):

```shell
cargo install cargo-dist
dist init   # generates a release workflow + [workspace.metadata.dist]
```

This complements release-plz: release-plz owns crates.io + version/changelog;
cargo-dist attaches binaries to the GitHub release.
