#!/usr/bin/env bash
# Verify nabu-core's public API surface against the committed snapshot.
#
# The nabu-core/src/lib.rs module split (see goals/breaking-monoliths/
# nabu-core-lib-split.md) turns lib.rs into a facade. Downstream crates compiling
# is necessary but NOT sufficient: a dropped or renamed `pub use` can still pass
# the build because no downstream file references every public symbol. This check
# is the load-bearing guard — it diffs the full public surface (under both the
# default and semantic feature sets) against the checked-in snapshot.
#
# Usage:
#   scripts/check-public-api.sh            # verify; non-zero exit on any diff
#   scripts/check-public-api.sh --update   # regenerate the committed snapshots
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
api_dir="$repo_root/crates/nabu-core/api"
default_snapshot="$api_dir/public-api-default.txt"
allfeatures_snapshot="$api_dir/public-api-all-features.txt"

if ! command -v cargo-public-api >/dev/null 2>&1; then
  echo "error: cargo-public-api not installed (cargo install cargo-public-api --locked)" >&2
  exit 2
fi

mkdir -p "$api_dir"

gen() {
  # $1: extra cargo flags  $2: output path
  cargo public-api -p nabu-core --simplified $1 >"$2" 2>/dev/null
}

if [[ "${1:-}" == "--update" ]]; then
  gen "" "$default_snapshot"
  gen "--all-features" "$allfeatures_snapshot"
  echo "updated snapshots in $api_dir"
  exit 0
fi

tmp_default="$(mktemp)"
tmp_all="$(mktemp)"
trap 'rm -f "$tmp_default" "$tmp_all"' EXIT

gen "" "$tmp_default"
gen "--all-features" "$tmp_all"

status=0
if ! diff -u "$default_snapshot" "$tmp_default"; then
  echo "PUBLIC API DRIFT (default features) — see diff above" >&2
  status=1
fi
if ! diff -u "$allfeatures_snapshot" "$tmp_all"; then
  echo "PUBLIC API DRIFT (--all-features) — see diff above" >&2
  status=1
fi

if [[ $status -eq 0 ]]; then
  echo "public API matches committed snapshot (default + --all-features)"
fi
exit $status
