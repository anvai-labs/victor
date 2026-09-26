# Pull requests and release workflow

Feature and maintenance changes target `develop`. Promotion pull requests target
`main`. Always specify the base branch: the repository default is `main`.

## Work in a linked worktree

Keep the main checkout on `main` or `develop`, and keep local configuration and
unrelated changes there. Create one worktree per pull request:

```bash
git fetch origin
git worktree add ../victor-my-task -b refactor/my-task origin/develop
cd ../victor-my-task
```

Keep the fewest active worktrees needed. A second worktree is appropriate for
independent work that can progress while another pull request awaits CI. Remove
each worktree and its local/remote branch after merge.

## Validate and open a pull request

Run targeted tests and the full affected suites, Black, Ruff, and MyPy on touched
production files. Before pushing, run the full collection check to catch stale
imports across the test matrix:

```bash
python -m pytest tests/ --collect-only -q
```

Before pushing a pull-request candidate, perform an adversarial review in a separate
review session. For concurrency, shared state, security or layering changes, exercise
the relevant negative paths explicitly. Reproduce findings and add positive and
negative regression coverage in the same pull request.

Record the completed review against the exact commit that was reviewed:

```bash
python scripts/adversarial_review.py record \
  --reviewer session-identifier \
  --summary "No open findings; exercised failure and boundary cases"
```

`pre-commit install` installs the pre-push hook. It rejects a branch commit without
a clean local attestation. Amending, rebasing, merging or adding a commit changes
the SHA and requires a fresh review. Attestations live under the repository's Git
metadata and are never committed.

This is a local process gate for participating clones, not a server-side security
control: it can be bypassed with `--no-verify`, and GitHub cannot observe the local
attestation. Server-side branch updates also bypass it. Repository auto-merge to
`develop` therefore relies on the required `CI Success` check; use a server-side
required check if centrally enforced adversarial review becomes necessary. Promotion
to `main` remains a separate, maintainer-controlled release operation.

Use conventional commit and PR titles such as `feat:`, `fix:`, `refactor:`,
`docs:`, or `ci:`. `release:` is not an accepted PR title type. Commit and PR text
must not contain agent attribution. The attribution checker permits the exact
Dependabot service trailer (`dependabot[bot]` with
`49699333+dependabot[bot]@users.noreply.github.com`); this text exception does not
authenticate authorship or exempt other bot identities.

```bash
git add path/to/changed-file
git commit -m "refactor: describe the resulting behavior"
git push -u origin refactor/my-task
gh pr create --base develop --head refactor/my-task \
  --title "refactor: describe the resulting behavior" --body-file /tmp/pr-body.md
```

Review the final pushed commit's checks. `CI Success` is the required aggregate:
it includes lint, types, import/boundary guards, changed-file tests, full-suite
collection, security checks, Rust packages, and native parity. Queued runners are
not test failures. See the [workflow gating map](https://github.com/anvai-labs/victor/blob/develop/.github/workflows/README.md).

## Minimize CI cycles

Hosted runners are shared across the organization. Complete the local change,
dependency resolution, affected tests, formatting/lint/typing, full collection and
required independent review before the first push. Batch compatible fixes and
related docs/version preparation into a reviewable candidate.

Stop before commit/push if a required local check fails or cannot run. Gate shell
automation on successful validation; never follow a failed test command with an
unconditional commit/push sequence. Recheck formatting and lint after the final
edit, including test fixtures. Security and architecture review findings belong
in the same locally validated candidate before its first CI-triggering push.

Routine Dependabot updates use a seven-day cooldown and grouped weekly PRs.
Urgent security fixes are triaged independently and do not wait for routine
version-update cooldowns. See the [GitHub cooldown documentation](https://docs.github.com/en/code-security/reference/supply-chain-security/dependabot-options-reference#cooldown).

Target one passing candidate cycle per PR and one promotion battery per closed
release batch. This is an efficiency target, never a reason to waive failures,
skip required checks, or delay an urgent security fix. Resolve all known failures
locally before a consolidated follow-up push. Re-run only failed jobs when there
is evidence of a transient infrastructure failure; changed code needs checks on
its new commit. Do not restart queued jobs or use remote CI as an edit/test loop.
Cancel only superseded runs belonging to this task, never unrelated work.

Keep the required aggregate reporting on every PR. Use shared scan reports and
compatible caches to remove duplicate work; test path filters and publication
prerequisites before changing them. Close the batch and inspect runner demand
before opening its promotion. Routine dependency updates are grouped; security
alerts receive prompt triage independently of the weekly update schedule.

## Merge and clean up

Merge only after the final commit's required checks and review pass. Squash merge
feature PRs into `develop`. An authorized maintainer may use `--admin`; this does
not replace verification of `CI Success`.

```bash
gh pr merge PR_NUMBER --squash
# Return to the main checkout before removing the worktree.
cd ../victor
git fetch origin
git worktree remove ../victor-my-task
# A squash-merged branch is not an ancestor: verify the merged PR before deleting it.
git branch -D refactor/my-task
# Delete the remote branch if GitHub has not already removed it.
git push origin --delete refactor/my-task
```

## Promote and release

Victor AI and `victor-contracts` use independent release trains. `VERSION` is the
AI version source; update it on `develop`, run `python scripts/sync_version.py
--ai`, and verify synchronization. The contracts package keeps its own version.

1. Open a `develop` → `main` promotion PR with an explicit `--base main`.
2. Wait for the promotion battery, including its sharded unit/integration,
   packaging, CLI smoke, and security checks. Feature-PR validation alone does
   not certify a release.
3. Merge the promotion using a **merge commit**, preserving ancestry.
4. Tag that main merge commit `vX.Y.Z`. `release.yml` publishes the AI package,
   native artifacts, Docker image, and GitHub Release.
5. Independently tag a contracts release `sdk-vX.Y.Z` to invoke
   `release-contracts.yml`.

The native wheel has an independent immutable version in `rust/pyproject.toml`;
update its Cargo manifest, Cargo lock and root native extra together. The version
check rejects a partial bump. Rebuilding changed native dependencies or metadata
requires a new version; publication does not silently reuse existing PyPI wheels.

A release run resolves its destinations before building. Production publication
requires a tag matching `VERSION`. For a rehearsal, dispatch with **Publish to
TestPyPI only**: builds and security checks still run, but production PyPI, Docker
and GitHub Release publication are disabled. This rehearsal publishes the core
package only; it does not publish native wheels to TestPyPI. Prerelease tags use TestPyPI and a
GitHub prerelease without replacing Docker `latest`. Tags use the package's
three-part version, optionally followed by a PEP 440 `aN`, `bN` or `rcN` suffix.

Release completion includes the extension and SBOM. All public assets are listed
in `checksums.txt` by their download filenames, including VSIX and SBOM files.
Download those assets together and run `sha256sum -c checksums.txt` (or
`shasum -a 256 -c checksums.txt` on macOS). Both fast CI and release lint read the
Ruff pin from the project's CI extra.

For an older squash-merged promotion, reconcile main's ancestry into develop
with a reviewed sync merge. Do not routinely rebase or force-push shared
integration branches. The September 2026 promotion sync used an `ours` merge only
after verifying that develop already contained the promoted content.

Framework API and architectural changes follow the [FEP process](../FEP_PROCESS.md).
The [proposal index](https://github.com/anvai-labs/victor/blob/develop/feps/README.md) records design status; merged design
documents do not by themselves establish implementation completion.

## CI gate map

The CI gate map groups actual jobs by responsibility. `CI Success` includes native parity;
failed or cancelled prerequisites fail the aggregate, while path-filtered skips are accepted.
The promotion battery is separate and runs on pull requests targeting `main`.

```mermaid
---
title: CI Success and promotion validation
---
%%{init: {"theme":"base","themeVariables":{"primaryColor":"#E8EFF7","primaryTextColor":"#17324D","primaryBorderColor":"#456987","lineColor":"#456987","fontFamily":"Arial"}}}%%
flowchart TB
  PR["Pull request / configured push"]
  subgraph FAST["ci-fast.yml"]
    N["Parallel prerequisite jobs<br/>format · lint · types · imports · docs<br/>boundaries · facade/hotspot · version/FEP<br/>quick tests · collection · security<br/>Rust packages · native-parity · PR metadata"]
    S["CI Success<br/>failed / cancelled prerequisite → failure<br/>passed / path-filtered skipped → success"]
    N -->|"aggregate every required need"| S
  end
  subgraph PROMO["Separate promotion battery · PR targeting main"]
    M["ci-test.yml matrix<br/>Python 3.12 / 3.13<br/>12 shards each · 24 jobs"]
    C["CLI Smoke Test"]
    T["Test Summary"]
    I["ci-integration.yml<br/>path-filtered integration suites"]
    M -->|"all shard results"| T
    C -->|"smoke result"| T
  end
  PR -->|"run fast gate"| N
  PR -->|"main target"| M
  PR -->|"main target"| C
  PR -->|"main target and matching paths"| I
```
