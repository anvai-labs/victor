# Releasing Victor

The [canonical PR and release workflow](../PR_WORKFLOW.md#promote-and-release)
owns branch protection, review, promotion and tagging. Prepare changes in a linked
worktree, merge them into `develop`, promote `develop` to `main` through the
complete promotion battery, then tag the verified main merge commit.

## Release Process Overview

### Independent versions

`VERSION` is the Victor AI version authority. After changing it, run
`python scripts/sync_version.py --ai` and
`python scripts/check_version_sync.py`. The synchronizer aligns the root package
and its exact native artifact version. Contracts, CodeGraph, first-party verticals
and Sandhi have independent versions and publication workflows.

Victor requires the exact `sandhi-gateway` binding pinned in `pyproject.toml`.
The separately operated Sandhi proxy is optional and has its own runtime identity.
The optional graph extra needs its exact published CodeGraph pin before Victor's
release can pass. Source or a GitHub release page alone does not prove required
wheels are available on supported Python versions, operating systems and CPUs.

## Prepare and promote

1. Fetch origin and reconcile open PRs, tags and the existing batch. Preserve
   unrelated work and create a linked worktree based on `origin/develop`.
2. Choose an unused version: patch for fixes, minor for compatible features.
   Update `VERSION`, synchronize AI/native metadata and update `CHANGELOG.md`.
   Change independent package versions only when they are in the qualified scope.
3. Run required tests, formatting, lint, configured typing, version checks,
   packaging and `python -m pytest tests/ --collect-only -q`. Collection is not
   execution. Complete independent adversarial review of the exact candidate and
   record the commit-bound attestation before the first push.
4. Open the preparation PR explicitly against `develop`. Merge after the final
   head's required checks and review pass. Close the batch before promotion;
   do not use hosted CI as a local edit/test loop.

Create the promotion with an explicit base:

```bash
gh pr create --base main --head develop \
  --title "chore: promote the prepared Victor release" \
  --body-file /path/to/reviewed-promotion-description.txt
gh pr checks PR_NUMBER --watch
```

Verify the final head and both `CI Success` and `Promotion CI Success`. The
promotion battery includes sharded tests, integration, CLI, packaging, security
and native qualification; feature-PR checks are insufficient. Merge the promotion
with a **merge commit**, preserving ancestry and main's protection requirements.

Fetch the merged result and verify its `VERSION` against the intended unused
tag. Create and push that one annotated tag on the verified main merge commit,
explicitly rather than using the current worktree's HEAD. Never move a published
tag. Direct pushes to main and `make release` do not replace protected promotion.

## Publication and artifact verification

The tag starts `.github/workflows/release.yml`. Consult its current jobs and
destinations rather than an older job inventory. It builds Python and portable
native wheels, standalone binaries, the container, VSIX and SBOM; checks the bare
installation and pinned graph dependency; creates checksums; and publishes
through its gates. See the [CI gate map](../PR_WORKFLOW.md#ci-gate-map).

A GitHub release appearing before all publishers finish is not completion.
Verify every intended destination and exact downloaded package version,
including both `victor-ai` and `victor-native`. Download release assets together
with `checksums.txt` and run:

```bash
sha256sum -c checksums.txt
# macOS: shasum -a 256 -c checksums.txt
```

Use the documented TestPyPI-only workflow input for an authorized rehearsal.
It does not publish native wheels or certify production publication. Reconcile a
failed publisher with immutable registry state before retrying. Do not replace
gates with an unreviewed manual upload or overwrite an existing version.

## Deployment and Victor–Sandhi alignment

Record source/tag SHA, package/native versions, executable paths, checksums,
configuration identity, service owner and rollback for each deployment. Inspect
the running gateway's `/version` and executable identity independently of the
Python binding. Compare that binding against Victor's exact pin.

Preserve established host roles. Keep a migrated, inactive gateway inactive unless
a topology change is requested. A new Victor release alone does not require
restarting an already-current Sandhi gateway.

1. Resolve immutable release artifacts and verify metadata/checksums before
   activation. Retain the previous installation and private configuration.
2. Install into an isolated environment first, with the matching native release
   when deployed and explicitly selected extras. From outside the source
   checkout, run `pip check`, `victor --version`, `victor --help` and
   `victor doctor`. Confirm imports resolve to the intended installation.
3. Check TLS, gateway readiness, model routes and credentials without printing
   secrets. Preserve token audiences, per-user attribution and the no-replay
   boundary for uncertain tool effects.
4. Refresh managed profiles with an authorized bootstrap token:

   ```bash
   victor profiles pull --gateway-url https://gateway.example
   victor profiles list
   ```

   Use the hidden prompt or supported environment variable, keeping tokens out of
   history/logs. Privately back up `~/.victor/profiles.d/sandhi.yaml` first.
   `~/.victor/profiles.yaml` overrides the central drop-in key by key; inspect
   the effective model, provider, endpoint, auth mode, context and output limits.
   A successful pull cannot remove stale user overrides. Do not copy another
   user's credentials between hosts.
5. Run bounded real complete and streaming calls on each intended provider, then
   a small tool task in a disposable directory. Verify the actual artifact,
   final response and persisted usage. Record unavailable routes separately;
   a hosted-provider pass does not qualify an unavailable inference service.
6. Activate the validated environment/service with recorded rollback. Confirm
   the active executable and repeat smoke tests through the real client path.
   Preserve sessions, state and credential stores.

See the [provider guide](../../user-guide/providers.md) and
[Sandhi co-design boundary](https://github.com/anvai-labs/sandhi/blob/develop/docs/td/TD-0008-victor-codesign-boundary.md).
Victor owns prompts, model/tool policy and pricing. Sandhi owns transport contracts
and neutral usage measurement. Gateway credential storage/use follows Sandhi's
operator model; client credential resolution follows Victor's configuration model.

## Checkpoint and recovery

Keep local tests, reviewed commits, merged source, published artifacts and live
acceptance separate. Record failures and executable next steps in the existing
workstream ledger or release evidence rather than duplicate status tables.
Remove merged task worktrees and branches; back-sync main into develop through
a reviewed sync PR when needed.

Hotfixes follow the same protected preparation, promotion, version and tag gates.
Do not directly merge into main or push all local tags. A partly published
version needs immutable-state reconciliation, not replacement artifacts.
