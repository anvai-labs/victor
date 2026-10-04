# Stage C reassessment — 2026-10-04

!!! info "Stage C remains partially implemented"

    Workflow-engine consolidation and the original docs rollout are complete.
    Runtime inversion, client routing and graph continuation still need focused
    execution. The retry capability follow-up removes the extra private probe
    introduced during gateway hardening; it does not complete FEP-0031.

**Verdict:** the old release-worktree reconciliation is complete; Stage C is not.
Item 23 and the original docs rollout are done. Item 27 is partial. The remaining
structural work must account for newer approval, retry and formation contracts.
This is a source/status audit, not a fresh certification of every runtime or live model.

## Release and publication boundary

| Surface | Verified state | Consequence |
| --- | --- | --- |
| Public release | [v0.12.0](https://github.com/anvai-labs/victor/releases/tag/v0.12.0), published 2026-10-04 | Plans to cut 0.9.2 or prepare an unpublished 0.10.0 are superseded. |
| Release evidence | [0.12.0 release run](https://github.com/anvai-labs/victor/actions/runs/37182750705) succeeded at `b9582b2fe` | The earlier 0.11.0 duplicate-upload failure is historical; artifact publication is not runtime acceptance. |
| Source identity | Develop `1c45c48ec` includes later lock refreshes and the optional TUI dependency split | Preserve normal promotion ancestry. The retry capability follow-up belongs to the next develop batch. |
| Changes since the September audit | Gateway identity/policy, result withholding, TLS diagnosis, Sandhi 0.10.1 accounting, interrupted-batch evidence, opt-in action observations, classify fallback, optional NumPy, Linux binaries, optional TUI dependencies and release/native version synchronization | These repairs refine the remaining backlog; they do not close all safety or formation acceptance gates. |
| GitHub Pages | [Main deployment](https://github.com/anvai-labs/victor/actions/runs/37182722213) succeeded 2026-10-04 at `b9582b2fe` | This reassessment reaches Pages through the next main promotion; PR builds validate without deploying. |
| Old release branches | Unique residual merged in [#1158](https://github.com/anvai-labs/victor/pull/1158); stale branches/worktrees retired with recovery archives | Do not reopen their superseded snapshots. |
| Separate active work | Dependency maintenance remains separate; #1209 action observations have merged | This audit does not certify security-alert closure or treat durable invocation observations as exactly-once execution. |

## Original backlog, reconciled

The [September review ledger](../reviews/2026-09-03-codesign/README.md#4-prioritized-maintenance-backlog)
retains historical findings. This table replaces the old handoff's execution assumptions.

| Item | Status | Current evidence and next acceptance condition |
| --- | --- | --- |
| 23 — ADR-030 / FEP-0007 | **Complete** | #1041–#1043 landed compiled adapters/streaming and removed the BFS walker and obsolete chat aliases. Keep engine and run/stream parity suites; do not repeat the migration. |
| 27 — FEP-0031 | **Partial** | Service-owned turn frame and enumerated planning, delivery, control, metrics, task/context, intelligence and provider-state categories are present. Six binding kwargs remain. Raw state and broad private accesses remain; factories, facade shims and proposed package/mixin work are not complete. |
| 32 — client routing | **Partial** | Client construction is present, but the REPL passes `agent` to `stream_response`; `/completions` obtains the orchestrator/provider and calls `provider.chat` directly. `ChatRequest` still has only `messages`; durable resume fields elsewhere are not a general session-aware client contract. |
| 24 — FEP-0032 | **Open; partial prerequisites** | Executor adapter exposes interrupt fields, but graph outcome/result do not. Ordinary interrupt-after checkpoints still resume the completed node. Sequential fan-out frontier metadata is a distinct partial repair and must be preserved. |
| 30 — coordinator split | **Open; rescope against current contracts** | `UnifiedTeamCoordinator` is 3,603 lines; merge execution and delegate-contract builders still live there. Proposed `WorktreeMergeService` / `DelegateContractBuilder` owners are absent. Preserve later isolation, accounting, pause and member-attribution fixes. |
| 31 → 29 — benchmarks / FEP-0033 | **Open** | BenchmarkRunner protocol and some conforming runners exist alongside high-level, code-generation and agentic runners. `victor/framework/rl` remains the implementation; `victor_contracts.rl_runtime` still points back to it; no `victor/rl` destination. Inventory live callers before consolidation, then invert contracts and relocate. |
| 28 — ADR-031 | **Open** | Contracts has vertical protocols/mixins, but no `verticals/bases` package. Extract small stable families with four-vertical behavior parity; do not treat domain-diverged files as literal duplicates. |
| 26b — graph indexing/storage | **Open** | `prepare_repository_snapshot` calls `parse_repo(root_path)` before filtering; parser still parses every recognized file. Stored hashes exist, but they do not establish parse reuse. `GraphStoreProtocol` remains combined rather than Tier A/B. Measure unchanged/one-file workloads before redesign. |
| 22a follow-up — vertical audit | **Open** | `contract_audit.py` retains its narrower forbidden-prefix tuple and deferral note. Recount/fix current violations before full-manifest migration; the old “28” is a historical count, not a fresh measurement. |
| D1–D3 — docs rollout | **Complete; ongoing drift maintenance** | Canonical docs index, diagrams, Material site and advisory built-site checker exist. Pipeline is `.github/workflows/docs.yml`, not the proposed `pages.yml`. Current landing/roadmap release labels needed correction. |

## Runtime inversion measurements

Measured using the repository's existing AST inventory. These conservative counts
include runtime-local accesses: **they are not a count of remaining facade leaks**.
The original “53 sites” design baseline is not the current measured total.

| File in `victor/agent/services/` | Private attributes | Private probes (cap) | Dynamic probes | Raw-state accesses |
| --- | ---: | ---: | ---: | ---: |
| `chat_stream_runtime.py` | 41 | 3 (3) | 0 | 5 |
| `chat_stream_executor.py` | 76 | 5 (5) | 4 | 0 |
| `chat_stream_helpers.py` | 56 | 13 (13) | 0 | 6 |
| `streaming_act_adapter.py` | 12 | 0 (0) | 0 | 0 |

All enumerated migrated access categories report zero. That is useful progress,
not proof that all FEP-0031 ownership targets are complete. Keep the FEP Draft
until its acceptance criteria are met; a merged design is not implementation completion.

## Reproduced interrupt/resume gap

A local, in-memory graph `a → b → END`, with `interrupt_after=["a"]`, increments a
counter in `a`. Invoke twice with the same thread and memory checkpointer:

| Observation | Current result | Required FEP-0032 behavior |
| --- | --- | --- |
| Node calls | `a, a` | Second invocation continues at `b`. |
| Counter | `1`, then `2` | Completed `a` is not executed again. |
| Result | Both `success=True`; no `interrupted` attribute | Explicit paused result distinct from completion. |
| Checkpoint | `node_id="a"` | Separate completed node from continuation target. |

No external tool or database was used. Existing HITL-controller tests do not
establish this graph-level contract. Implement and test ordinary edges, terminal
interrupts, conditional routing, sequential fan-out and supported dynamic fan-out
before claiming full resume parity; retain checkpoint deep-copy guarantees.

## Revised execution order

| Order | Bounded next work | Exit gate |
| --- | --- | --- |
| Done | Bind provider retry ownership to the existing lifecycle capability and explicitly select retry/boundary CI suites | Probe cap returns 14 → 13; gateway attempts remain single, direct retries bounded, policy errors terminal. |
| 2 | Reconcile newer safety work with remaining G60–G70 and G72 findings | Map #1180/#1185/#1188/#1192/#1203/#1204/#1208/#1209 to exact repaired paths; preserve unresolved principal/session ownership, sibling control signals, durable effects, admission, budget and retrieval gaps. |
| 3 | Complete FEP-0031 ownership slices and item 32 client routing | Service capabilities own behavior; retire real callers before removing shims; keep session identity, approval, cancellation and accounting parity. |
| 4 | Implement FEP-0032 graph pause/continuation semantics | Negative replay tests and explicit paused results; align with existing approval stores without conflating them. |
| 5 | Extract coordinator services, then benchmark consolidation → contracts-first RL relocation | Preserve current formation contracts; establish one measured benchmark/session path before moving consumers. |
| 6 | Vertical family extraction, full boundary manifest, and measured graph-indexing work | Contract parity, no import-boundary widening, unchanged-file parse reuse measured against baseline. |
| Release | Close a bounded batch and verify main/develop ancestry and release scope | One promotion battery; artifact checks and Pages publication. No new release is implied by this audit. |

Use the [safety audit](agentic-workflow-safety-audit.md) and
[formation evidence ledger](multiagent-formation-coverage-handoff.md) as the owners
of G60–G72 and C5 details. The newer #1204 result-publication repair is bounded: raw executor/observer disclosure
and cached pre-action authorization remain open. #1206 repairs metadata-only streaming
usage accounting (G71); it does not redeploy the gateway or establish C5 acceptance.
#1208 adds interrupted-batch evidence and blocks unsafe resume; it does not establish
general graph continuation or live C5 acceptance. #1209 persists opt-in single-action
intent/observations, blocks replay after intent and requires strict result publication.
Backend receipt reconciliation and whole-member continuation remain open. G72 records
permissive legacy supervisor selection/delegation as separate unfinished work. Retained failures and operational
gates are not closed by source fixes, passing unit tests, or the 0.12.0 release. This
reassessment does not run another live model/formation battery.

## Evidence and verification

- Source baseline: `1c45c48ec` plus the provider-retry capability follow-up (#1237); publication state checked 2026-10-04.
- The September `6d3b7e14d` audit found **88 passed, 1 failed** in boundary/hotspot checks (14 probes versus cap 13). #1215 raised the cap to 14; the capability follow-up restores 13 by removing the private lookup.
- Capability validation: **1,347 affected tests**, **185 selected tests**, **100% changed production-line coverage**, and **34,047 tests collected**; formatting, lint and typing passed. These are local checks, not live-model acceptance.
- In-memory graph observation above; source: `graph_runtime.py`, `graph_execution.py`, `graph_checkpoint.py`. Later release changes do not modify those paths.
- Source inspected: workflow adapters/streaming, chat service/bindings, CLI REPL, API chat/completions routes, graph parser/store, coordinator, benchmark runners, RL bridge and vertical contract tree.
- Documentation checks: strict MkDocs build, built-site links, docs drift and repository hygiene. Six existing safety-audit anchors and two FEP links outside the MkDocs tree are corrected in this update.
- Independent source review confirmed the original inventory and reproduced its boundary failure; the retry follow-up receives separate adversarial review.
