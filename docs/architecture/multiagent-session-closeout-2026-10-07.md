# Multiagent session closeout and restart handoff

**Snapshot: 2026-10-07.** This is the durable restart index for the Victor formation,
approval and recovery session. It records source and historical evidence, not a new
live acceptance run. The [formation ledger](multiagent-formation-coverage-handoff.md)
owns coverage/gaps; the [safety audit](agentic-workflow-safety-audit.md) owns the design
critique; the [roadmap](../roadmap.md) owns priority. Do not restart WS-A from scratch.

## Current continuation entry point

The [shared agent-service tracker](victor-agent-service-plan.md) now owns the
cross-session execution order, stable VAS task IDs, TDD/smoke gates and checkpoint
updates. Read it with FEP-0037 and reconcile PR/source state before acting. This
October 7 handoff preserves history; its priority list below is context, not a
second current task-status ledger. Original WS-A–WS-I stay landed and C5 stays
open until its own acceptance gate passes.

## Completed milestones and limits

| Milestone | Durable record | What is still outside that milestone |
| --- | --- | --- |
| Original WS-A–WS-I increments | 9/9 landed; PR links in the formation ledger §4 | Matching model/task cohorts and lifecycle acceptance have separate denominators |
| Exact single-agent approval and interrupted tool batches | FEP-0029; #1185/#1208 | Authenticated principal/member ownership and complete member continuation |
| Opt-in single-action intent/observations | [#1209](https://github.com/anvai-labs/victor/pull/1209), merged October 3 as `bc3e7c3f9`; included in v0.12.0 | `returned` is an invocation observation, not a verified business receipt; `pending`/`unknown` never authorizes replay |
| Haskell grammar compatibility | Included in #1209; alias query handles both grammar spellings and captures only the defined name | No new formation or model-quality claim |
| Async journal and transaction cleanup | [#1249](https://github.com/anvai-labs/victor/pull/1249), main `f9515e98e`, October 7 | Post-v0.12.0 source; backend receipt lookup and C5 remain open |
| Packaging release | [v0.12.0](https://github.com/anvai-labs/victor/releases/tag/v0.12.0), published October 4 | A release tag does not prove the currently running binaries/configuration or subsequent source fixes |
| Documentation publication | [Pages run 37570194010](https://github.com/anvai-labs/victor/actions/runs/37570194010), main `f9515e98e`, successful October 7 | This closeout needs a later reviewed main promotion to appear on Pages |

No new live calls were made during closeout. Historical ZAI cohorts passed 15/15;
the September 23 Qwen3 single-file cohort was interrupted (0/15 accepted), and the
matched Qwen14 cohort remained unstarted. These are workload-specific observations,
not proof of insufficient model capacity. C5 / [InferFlux #184](https://github.com/anvai-labs/inferflux/issues/184)
was verified **OPEN** on October 7. Keep original failed evidence.

## Integration and validation provenance

At entry, `origin/develop` was `339b1d40c` and `origin/main` was `f9515e98e`.
Main contained all of develop plus the exact gateway 0.11.0 pin/lock updates (#1241)
and asynchronous durable-action follow-up (#1249). The closeout candidate starts
from develop and incorporates that existing main content, with no new runtime
behavior beyond those already merged changes.

The historical [#1249 full-suite run](https://github.com/anvai-labs/victor/actions/runs/37565494594)
has failures in shards 4 and 8 on both Python 3.12 and 3.13. Its green fast aggregate
does not turn those failures into a full pass. Both failures reproduced locally:

- `test_ab_list_experiments_uses_global_victor_dir_by_default` patched an old eager
  import. The repair patches the canonical lazily imported paths function while
  retaining the real SQLite fixture and output assertions.
- `test_publication_jobs_consume_the_release_plan` found a missing
  `if-no-files-found: error` on the distribution-metrics artifact upload. The repair
  restores that release gate; the existing contract test remains unchanged.

The existing tests already cover these contracts, so no parallel or duplicate suite
was added. Candidate checks and the reviewed SHA belong in the closeout PR; do not
claim a passing rerun of historical shards from targeted tests. A merged closeout
does not by itself publish a new release or promote documentation to main.

## VS Code dependency follow-up

[#1250](https://github.com/anvai-labs/victor/pull/1250) merged into develop as
`b8e195cc68e009c68b2cfc9b96ac8cf5b74bef84`. Its branch and worktree were removed.
The [VS Code checkpoint](../development/vscode-dependency-validation.md) records
Node 24.21.0 LTS, repaired dependency graphs, actual activation smoke coverage,
and six pruned inactive command advertisements. Source, default-branch alert
closure, published artifacts and running services remain separate milestones.
The gateway/origin and G62/G61/G70/C5 follow-ups below remain open.

## Next PR-sized milestones, in order

1. **Foundation/integration gate.** Verify the closeout PR merged and fetch both
   branches. Preserve #1249 and #1241 when reconciling promotion ancestry. Confirm
   Sandhi/InferFlux release, source, binary and config identities; discover actual
   routes and ready models after restart. Sandhi 0.11.0 already includes admission,
   terminal observations, settlement/recovery inventory and proxy ownership source
   increments (#308–#316); audit the remaining deployment/HTTP/lifecycle acceptance
   against that release rather than implementing those increments again. Recheck
   their handoffs; old process names and September liveness
   evidence are not current deployment acceptance.
   Preserve #1249's recorded follow-ups: resume/claim and turn-boundary pause store
   calls still need an off-loop audit, and contention needs a loop-responsiveness
   regression. The journal change alone does not establish a fully nonblocking lifecycle.
2. **G62 verified external-action recovery.** Extend the existing paused-run/action
   owner with typed backend lookup or explicitly guaranteed same-key deduplication.
   Persist receipts and unknown/confirmed status. Acceptance: backend commits then
   loses its response; process restarts; duplicate events cannot create a second
   effect; a verified receipt resolves uncertainty. Without backend support, retain
   unknown status and require intervention. No second registry or dispatcher.
3. **G61/G70 complete member continuation.** Bind caller/member, exact payload and
   relevant versions; recheck authority before dispatch. Restore completed prior
   batches/turns and reconcile uncertain effects before advancing. Test stale
   approval, owner mismatch, cancellation and crashes across checkpoints in the
   existing resume/member owners. G60 publication/disclosure and G63 ownership gates
   remain applicable; do not bypass them to obtain a green formation run.
4. **C5 acceptance.** On accepted released foundations, run
   `scripts/validation/multiagent_gateway_live.py --mixed` with the established
   six-Qwen/one-ZAI workload. Keep the confirmed 120-second buffered deadline unless
   an explicitly versioned acceptance plan changes it; count timeouts as failures.
   Require deliverables, passing task pytest, seven distinct member sessions,
   request correlation and wire/SQLite/C4/dashboard conservation. Record exact
   runtime identities and remaining tokenizer/reuse/session/cancellation limits on
   #184; close C5 only after its full verdict is reviewed.
5. **Matched formations and bounded semantics.** Preserve the ZAI reference; finish
   explicitly labelled simpler local-model cohorts without weakening their oracle.
   Choose a different local model only after measuring failures/capacity. G72 strict
   supervisor selection/delegation is a separate opt-in increment. A single agent
   need not be a supervisor, and peer formations need not route through one.

Every increment uses the §2.4 structured contracts, one dispatch/registry, one
identifier derivation and unchanged defaults. Use TDD in existing test owners,
review duplicate coverage before adding tests, and keep feature PRs targeting
develop with no force-push. Do not substitute model confidence for verified outcomes.

## Dependency checkpoint

Upstream was fetched again during closeout. GitHub and the Python package index
both identify **Sandhi 0.11.0** as the latest published release on October 7. The
candidate incorporates #1241's exact `sandhi-gateway==0.11.0` requirement in
`pyproject.toml`, `requirements.txt`, and the API/CPU-embedding deployment locks.
No speculative version bump or unreviewed development dependency is needed.
The historical deployment handoff's Sandhi 0.12.0 references were inaccurate;
Victor's own release is 0.12.0, with an independent version train.

The latest published InferFlux release is
[0.4.0](https://github.com/anvai-labs/inferflux/releases/tag/v0.4.0), September 25.
InferFlux is a separately deployed service, not a Victor Python dependency. Its
release tag is not proof that a local or aiserver1 process runs that binary. Verify
source/binary/config identities before acceptance or a cutover. Unrelated
dependencies retain the reviewed #1242 lock refresh; this is not an assertion that
every transitive dependency is the newest version available.

## Reboot and resume

The old `/private/tmp/victor-action-reconciliation` worktree was removed after
#1209 merged. Residual directories are not an active Git checkout. The source,
tests and contract are in Git; do not recover or replay the obsolete branch.
Tracked evidence under `docs/architecture/evidence/`, the linked PR/run records,
and this handoff are the durable record; temporary logs/venvs are disposable.
The older #1209 PR body cites pre-final local validation and is not an attestation
for every later commit. Use the actual merge SHA and current checks.

At the read-only October 7 process check, this Mac still had two Sandhi proxy
processes (under `var/sandhi-zai/` and `var/sandhi-oidc/`), an SSH process, and an
InferFlux executable under `/tmp/inferflux-build/`. Their live configuration,
ownership, readiness and restart supervision were **not** established. No service,
cache or credential was changed. Other sessions remain active; this closeout only
establishes preservation of this Victor work, not machine-wide safe shutdown.

After reboot:

1. In `~/code/codingagent`, run `git fetch origin --prune`, `git status --short
   --branch` and `git worktree list`. Read this file from `origin/develop`; #1250 is merged. Verify PR state
   before deleting a worktree. Do not alter another session's locked checkout.
2. Create the next linked worktree from the reconciled `origin/develop`. Recreate
   `.venv-codesign`, install in-repo contracts before runtime dependencies, and use
   `.venv-codesign/bin/python -m pytest`; never bare pytest. Disable optional MLX
   probing with `VICTOR_ENABLE_MLX_PROVIDER=0` for these unrelated control tests.
3. Reuse private local credentials without printing or transferring them. Inspect
   current service configuration before restoring tunnels; do not blindly reuse
   the obsolete 8081/18081 recipes. InferFlux placement belongs to InferFlux, while
   Victor selects model IDs and Sandhi supplies optional transport/accounting.
4. Start at milestone 1 above. Do not rerun the passing five-call replay merely to
   regenerate evidence; no shared-cache clearing and no synthetic probe presented
   as actual-member acceptance.

Suggested session prompt:

> Read docs/architecture/multiagent-session-closeout-2026-10-07.md, then the canonical
> formation ledger and safety audit. Verify merged PRs and main/develop ancestry.
> Continue the next incomplete gate in order: released foundations, G62 receipts,
> member continuation, then C5. Preserve failed evidence and credentials; do not
> count source merges or startup probes as live acceptance.
