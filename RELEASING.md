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

Releases are driven by you, not a bot. You bump the version and edit the
changelog in a normal PR, then trigger the release from the Actions tab. There
is no auto-opened release PR and no PAT.

## One-time setup

1. Create a crates.io API token at <https://crates.io/settings/tokens>. Scope the
   CI token to publish only.
2. Add it as a repository secret named `CARGO_REGISTRY_TOKEN` (Settings →
   Secrets and variables → Actions, or `gh secret set CARGO_REGISTRY_TOKEN`).
   **Never commit the token** — not in `Cargo.toml`, the workflow, or any
   tracked file.

No other secret is required. The release workflow tags the commit and creates
the GitHub release with the built-in `GITHUB_TOKEN` (`contents: write`).

## Cutting a release

1. **Open a version-bump PR.**
   - Set `[workspace.package].version` in the root `Cargo.toml` to the new
     version (e.g. `0.1.1`). All four crates inherit it.
   - In `CHANGELOG.md`, rename the `## Unreleased` heading to `## <version>`
     and start a fresh empty `## Unreleased` above it. The release notes are
     taken verbatim from the `## <version>` section.
   - Open the PR, let CI pass, merge to `main`.

2. **Run the release workflow.** Actions → **release** → **Run workflow**.
   - Leave `publish` checked to publish for real.
   - Uncheck `publish` for a dry run: it runs the version/changelog checks and
     the full test gate but publishes nothing and creates no tag.

The workflow then:

- **preflight** — reads the version from `Cargo.toml`, extracts the matching
  `## <version>` section from `CHANGELOG.md`, and fails if the `v<version>` tag
  already exists or the changelog section is missing.
- **gate** — `cargo fmt --check`, `cargo clippy -D warnings`, and
  `cargo test --workspace` with and without the `semantic` feature, against the
  exact commit being released.
- **release** (only when `publish` is checked) — `cargo publish` for each crate
  in dependency order (cargo waits for each to appear in the index before the
  next resolves), then tags `v<version>` and creates the GitHub release from the
  changelog section.

Re-running for an already-released version fails closed: preflight stops on the
existing tag, and `cargo publish` rejects a version that already exists on
crates.io.

Verify in a clean environment: `cargo install nabu-cli` installs the `nabu`
binary.

## First release (already done; for reference)

The first publish claims the crate names and is irreversible (you can only
`yank`, never delete). It was done in dependency order with dry-runs first. To
reproduce the manual dance from a clean checkout of `main`:

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
# now-published upstream:
cargo publish -p nabu-core
cargo publish -p nabu-adapters
cargo publish -p nabu-mcp
cargo publish -p nabu-cli
```

After the first release, use the workflow above instead of publishing by hand.

## Optional: prebuilt binaries (cargo-dist)

crates.io requires users to have a Rust toolchain. To also ship prebuilt
binaries on GitHub Releases (macOS/Linux/Windows, no local compile of the
semantic/ONNX stack), add [cargo-dist](https://opensource.axo.dev/cargo-dist/):

```shell
cargo install cargo-dist
dist init   # generates a release workflow + [workspace.metadata.dist]
```

This complements the release workflow: the workflow owns crates.io + the
version/changelog + the GitHub release; cargo-dist attaches binaries to that
release.
