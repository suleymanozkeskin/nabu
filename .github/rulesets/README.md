# Repository rulesets

Version-controlled definitions of this repo's GitHub rulesets. GitHub does not
apply these files automatically — they are the source of truth, applied with
`gh`.

## Apply (create)

```shell
gh api -X POST repos/suleymanozkeskin/nabu/rulesets \
  --input .github/rulesets/main-branch.json
gh api -X POST repos/suleymanozkeskin/nabu/rulesets \
  --input .github/rulesets/release-tags.json
```

## Update an existing ruleset

```shell
gh api repos/suleymanozkeskin/nabu/rulesets --jq '.[] | "\(.id)\t\(.name)"'
gh api -X PUT repos/suleymanozkeskin/nabu/rulesets/<id> \
  --input .github/rulesets/main-branch.json
```

## What they enforce

- **`main-branch.json`** — the default branch requires a pull request (0
  approvals, solo-friendly) and the CI checks `fmt + clippy`,
  `test (ubuntu-latest)`, and `test (macos-latest)` before merge; force-pushes
  and deletion are blocked.
- **`release-tags.json`** — `v*` and `nabu-*` release tags cannot be deleted or
  force-updated.

## Notes

- Status-check `context` values must match the CI job names exactly. If a job in
  `.github/workflows/ci.yml` is renamed, update `main-branch.json` and re-apply.
- `semantic-acceptance` is intentionally **not** a required check — it runs on a
  schedule / `workflow_dispatch`, not on pull requests, so it never reports a
  status on a PR.
- release-plz opens its release PR with the built-in `GITHUB_TOKEN`, and PRs
  opened by `GITHUB_TOKEN` do not trigger workflows. With the required checks
  above, that PR would never get a CI status and could not merge. Give release-plz
  a PAT stored as `RELEASE_PLZ_TOKEN` so its PRs run CI — see
  [`RELEASING.md`](../../RELEASING.md).
