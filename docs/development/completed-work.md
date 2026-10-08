# Completed and superseded work

This compact record preserves outcomes and failed evidence from retired interim
notes. It is not a second backlog. Use the [repository map](repository-map.md) for
code/document owners, the [roadmap](../roadmap.md) for broader priorities and the
[agent-service tracker](../architecture/victor-agent-service-plan.md) for current
VAS status. Original records remain in immutable Git history below.

## Verified milestones and limitations

| Work | Evidence retained | Current owner / remaining boundary |
| --- | --- | --- |
| Vertical test relocation (March 2026) | Commits `a2b4f6999`, `5882636d1`, `1490a0f66`; historical runs: **1,545 passed / 1 failed**, **30 passed / 5 skipped**. These are not full-green results. | [Vertical packages](https://github.com/anvai-labs/victor/tree/develop/verticals), [test strategy](testing/strategy.md), [contract boundary](../architecture/CONTRACTS_BOUNDARY.md). Old external-repository migration steps are superseded by the monorepo; test absence alone is not completion. |
| Coordinator consolidation | Historical phases 1–3 consolidated strategy dispatch/types; current owner is `UnifiedTeamCoordinator`. Earlier “no known issues” and compatibility/removal promises are not current guarantees. | [Team guide](../guides/MULTI_AGENT_TEAMS.md), [migration guide](https://github.com/anvai-labs/victor/blob/develop/victor/teams/MIGRATION_GUIDE.md), formation ledger G61/G62/G70. |
| Runtime component refactoring | The historical SOLID report mixed real components with an unwired construction proposal. F-017 / [#560](https://github.com/anvai-labs/victor/pull/560) removed the dead bundle layer; it was never the live construction path. | [Service architecture](../architecture.md), [review backlog](../architecture/CODEBASE_REVIEW_BACKLOG.md). Do not reinstate obsolete `create_all_components` examples. |
| Workflow engine consolidation | [ADR-030](../architecture/adr/030-single-graph-execution-engine.md), #1041–#1043 record the engine migration. The historical plan's unchecked criteria are not new completion evidence. | ADR-030 and maintained [workflow examples](../guides/workflow-development/examples.md); FEP-0032 interrupt/resume remains separate. |
| Documentation consolidation (September 2026) | Original audit: **320 tracked files**, 260 Markdown; provisional **237 CURRENT / 83 STALE-but-keep / 0 DEAD**. It scanned all local links but performed targeted semantic review; no claim of complete API/example certification. Later source/link/index fixes are in its immutable record. | [Publishing](docs-publishing.md) and [repository map](repository-map.md). October cleanup makes a new, explicit supersession decision; it does not rewrite the original zero-DEAD finding. |
| Packaging phases 0/1 | v0.12.0 work: #1224–#1227, #1230–#1234, #1238–#1239, #1242–#1243, #1247. Explicit click dependency; numpy-free core fallbacks; uv-first installation; native/frozen artifacts and smoke gates; Textual optional. | [ADR-016](../architecture/adr/016-distribution-packaging-strategy.md), [release procedure](releasing/publishing.md), [remaining packaging checks](dependencies.md#packaging-follow-ups). Historical phase-6 analysis is superseded, not fully implemented: no assumed `victor-ai-core` split, blanket stdlib replacement or speculative speedup mandate. |
| October 7 main/develop integration | [#1250](https://github.com/anvai-labs/victor/pull/1250), `b8e195cc68e009c68b2cfc9b96ac8cf5b74bef84`, preserved #1241 gateway pin and #1249 journal changes. [Historical #1249 run](https://github.com/anvai-labs/victor/actions/runs/37565494594) failed shards 4/8 on both Python versions. Lazy-path patching and missing artifact-upload failure policy were repaired through existing tests; targeted passes do not retroactively pass those shards. | VAS tracker; Sandhi pin **0.11.0** is independent of Victor **0.12.0**. Deployment identities still require live verification. |
| VS Code dependencies and activation | [#1251](https://github.com/anvai-labs/victor/pull/1251): Node 24.21.0, both npm audits zero at validation, VSCE 4, 999 host / 50 unit / 83 repository tests, 16-file VSIX. Stronger activation first failed on duplicate symbol registration, then six inactive command advertisements; both repaired. Whole-extension coverage was only 5.35% statements / 5.43% lines; nine lint and bundle-size warnings remained. | VAS-02/13/20. Main promotion/rescan and published-extension acceptance are separate. Do not restore six dormant commands without ownership-safe integration; dormant storage duplicates the active `exportConversation` command. |
| Existing client streaming compatibility | [#1252](https://github.com/anvai-labs/victor/pull/1252) repaired the actual TS→core HTTP 422 request mismatch and web settings import. Activation alone had not proved chat worked. | VAS-03a complete; shared API/identity/EOF/cancellation contracts remain with their VAS owners. |
| Durable action recovery | [#1209](https://github.com/anvai-labs/victor/pull/1209) single-action observations; [#1249](https://github.com/anvai-labs/victor/pull/1249) async journal; [#1253](https://github.com/anvai-labs/victor/pull/1253) off-loop approval admission; [#1255](https://github.com/anvai-labs/victor/pull/1255) bound backend receipts. | VAS-11c/12/17/18: production adapter qualification, recovered-result publication, whole-member continuation and released mixed-team C5. Receipt verification does not replay a tool or continue a member. |
| Formation implementation and live cohorts | Original WS-A–WS-I **9/9 landed**. Historical OIDC ZAI cohorts **15/15**; September 23 Qwen3 cohort interrupted, **0/15 accepted**; Qwen14 unstarted. Failed and interrupted cohorts are retained. | [Formation ledger](../architecture/multiagent-formation-coverage-handoff.md), InferFlux #184 / VAS-18. No new provider, release or C5 evidence in this cleanup. |

## Validation provenance

The [original VAS checkpoint history](https://github.com/anvai-labs/victor/blob/76384b0e455bcaea29c4046a15ff0092cfe6df1d/docs/architecture/victor-agent-service-plan.md)
retains complete commands, intermediate counts and initial failures. The active
tracker now keeps current state, acceptance contracts and the latest verified
milestone instead of stale claim/push instructions.

| Increment | Preserve this failure / final evidence / limitation |
| --- | --- |
| #1251 | First hosted FEP validation failed on missing canonical sections, then corrected. Local: 999 host, 50 unit, 83 repository checks; 33,883 collected. Merge gates passed on the corrected head, not the initial candidate. |
| #1252 | Compiled TS→HTTP returned 422 before repair. Initial quick CI failed because the dev extra lacked Uvicorn and the pytest executable omitted the source-only web namespace from its import path. Corrected dependencies/`python -m pytest`, kept real HTTP assertions; 96 corrected HTTP/config/CI tests passed. Final merge: 41 applicable green checks, independent review clean. Low whole-extension coverage and failed canvas/OIDC demonstration remain separate. |
| #1253 | Real SQLite contention blocked the event loop until a five-second watchdog released the lock before repair; changed-test mapping also failed before correction. Final CI-selected suite 174 passed, broader affected suite 216, 33,894 collected; 39 successful merge checks and exact-head clean review. Cancellation may consume a claim in a late worker without dispatch; consumed-but-undispatched recovery remains open. Injected custom stores retain their threading/responsiveness responsibility. |
| #1255 | Five receipt tests failed before implementation; independent review reproduced unsafe extended-receipt persistence and malformed/downgraded disclosure. Corrected both in existing test owners. Final 580 affected / 221 boundary checks (8 optional skips), 180 independently rerun, 33,946 collected, 93% changed-line coverage; 40 applicable checks and clean exact-head review. No production adapter, transcript publication, continuation, release or live C5 acceptance. |

Original failed and passing local evidence is also indexed by the current VAS
tracker's durable archive paths. Archive availability is machine-local; merged
Git/PR/run identities are the cross-machine record. No credentials or private
request payloads belong in these records.

## Retired interim records

Removed from the active tree on October 8, 2026 after cross-referencing the current
owners. These links preserve the entire former file, including superseded ideas
and original limitations. They are provenance, not instructions to execute.

| Former path | Disposition / replacement |
| --- | --- |
| [docs/architecture/deployment-reshaping-handoff.md](https://github.com/anvai-labs/victor/blob/76384b0e455bcaea29c4046a15ff0092cfe6df1d/docs/architecture/deployment-reshaping-handoff.md) | Completed packaging record above; open packaging tasks transferred to dependencies guide; unsafe/stale release commands replaced by maintained release procedure. |
| [docs/architecture/multiagent-session-closeout-2026-10-07.md](https://github.com/anvai-labs/victor/blob/76384b0e455bcaea29c4046a15ff0092cfe6df1d/docs/architecture/multiagent-session-closeout-2026-10-07.md) | Historical source/failed-CI/live limits preserved above; VAS tracker owns restart and current priorities. |
| [docs/architecture/phase6-packaging-plan.md](https://github.com/anvai-labs/victor/blob/76384b0e455bcaea29c4046a15ff0092cfe6df1d/docs/architecture/phase6-packaging-plan.md) | Superseded analysis; ADR-016 and measured native strategy govern current decisions. Unimplemented proposals are not marked complete. |
| [docs/architecture/workflow-consolidation-plan-historical.md](https://github.com/anvai-labs/victor/blob/76384b0e455bcaea29c4046a15ff0092cfe6df1d/docs/architecture/workflow-consolidation-plan-historical.md) | Superseded engine proposal; ADR-030 owns delivered engine work, FEP-0032 owns outstanding resume design. |
| [docs/development/testing/all-verticals-migration-plan.md](https://github.com/anvai-labs/victor/blob/76384b0e455bcaea29c4046a15ff0092cfe6df1d/docs/development/testing/all-verticals-migration-plan.md) | Superseded external-repo procedure; historical relocation evidence above and current monorepo test owners. |
| [docs/development/testing/category-5-analysis.md](https://github.com/anvai-labs/victor/blob/76384b0e455bcaea29c4046a15ff0092cfe6df1d/docs/development/testing/category-5-analysis.md) | Interim relocation list consolidated with the verified migration commits above. |
| [docs/development/testing/vertical-migration-summary-2026-03-04.md](https://github.com/anvai-labs/victor/blob/76384b0e455bcaea29c4046a15ff0092cfe6df1d/docs/development/testing/vertical-migration-summary-2026-03-04.md) | Completed relocation record condensed above with failed/skipped evidence unchanged. |
| [docs/development/docs-audit-2026-09.md](https://github.com/anvai-labs/victor/blob/76384b0e455bcaea29c4046a15ff0092cfe6df1d/docs/development/docs-audit-2026-09.md) | Provisional historical inventory condensed above; repository map owns current discovery. |
| [docs/development/vscode-dependency-validation.md](https://github.com/anvai-labs/victor/blob/76384b0e455bcaea29c4046a15ff0092cfe6df1d/docs/development/vscode-dependency-validation.md) | Merged #1251 evidence condensed above; current dependencies/smoke contract remain in VS Code sources and VAS tracker. |
| [victor/agent/SOLID_REFACTORING.md](https://github.com/anvai-labs/victor/blob/76384b0e455bcaea29c4046a15ff0092cfe6df1d/victor/agent/SOLID_REFACTORING.md) | Historical/partly aspirational report; F-017 and current services supersede unwired construction examples. |
| [victor/teams/CONSOLIDATION.md](https://github.com/anvai-labs/victor/blob/76384b0e455bcaea29c4046a15ff0092cfe6df1d/victor/teams/CONSOLIDATION.md) | Overlapping historical phases 1–3 record; current team guide and migration guide remain. |
| [victor/teams/RELEASE_NOTES.md](https://github.com/anvai-labs/victor/blob/76384b0e455bcaea29c4046a15ff0092cfe6df1d/victor/teams/RELEASE_NOTES.md) | Duplicate historical consolidation narrative; root release history and current team guide remain. |
