#!/bin/sh
# Point this clone's git hooks at the tracked .githooks directory. Hooks are not
# shared automatically; each clone must opt in once by running this script.
set -eu

repo_root=$(git rev-parse --show-toplevel)
git -C "$repo_root" config core.hooksPath .githooks
chmod +x "$repo_root"/.githooks/* 2>/dev/null || true

echo "git hooks enabled: core.hooksPath -> .githooks"
echo "  pre-commit: cargo fmt --all --check"
echo "  pre-push:   cargo clippy --workspace --all-targets -- -D warnings"
