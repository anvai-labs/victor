# Multi-Agent Formation Coverage, Gaps, and Handoff

**Date:** 2026-09-17 · **Status:** Handoff for a follow-up session · **Live matrix:** R9700 / InferFlux / Qwen3-Coder-30B (see [multiagent-formations-inferflux.md](multiagent-formations-inferflux.md))

This document answers three questions: which formations Victor implements and which of
those were verified live; which designs exist beyond the canonical six (implemented,
unwired, or deferred); and what the follow-up session should address, with the
co-design learnings that motivate each item.

## 1. Coverage: what is implemented, and what was verified live

### 1.1 Canonical formations — implemented and live-tested (6/6)

`TeamFormation` ([`victor/teams/types.py`](../../victor/teams/types.py)) defines exactly six
values, all registered in `UnifiedTeamCoordinator._formations` and all verified live on
2026-09-17 (single 3-member PARALLEL re-verified again on rebased develop, 48.2s green):

| Formation | Live result | Notes from the run |
|---|---|---|
| SEQUENTIAL | ✅ 30.0s | context chains member→member |
| PIPELINE | ✅ 18.0s | both stages delivered files |
| PARALLEL | ✅ 44.7s / 48.2s | 3 concurrent members, disjoint deliverables |
| HIERARCHICAL | ✅ 19.0s | supervisor delegation + synthesis |
| CONSENSUS | ✅ 18.5s | agreement reached in one pass |
| REFLECTION | ✅ 48.2s | generator→critic→refine, early exit |

### 1.2 Implemented, real user surface — NOT live-tested

These shipped and are unit-tested, but the live InferFlux matrix did not exercise them:

- **Dynamic formation selection.** `StateGraphNodeConfig.formation_strategy` picks a
  formation per invocation from graph state (sync-only; async strategies raise
  `TypeError`), and `formation_hint` / `topology_formation_hint` context keys override
  per call (`_resolve_effective_formation`, unified_coordinator.py). A raising strategy
  falls back to the default formation — logged at **DEBUG only**.
- **`max_workers` clamping** (`_extract_max_workers` / `_limit_execution_members`) —
  bounds concurrent member execution; untested against a capacity-limited server.
- **Worktree-isolated members.** `victor/teams/worktree_runtime.py`
  (`WorktreeAssignment`, `WorktreeExecutionPlan`) plus `worktree_planner`/`worktree_runtime`
  coordinator params and colon-format `child_session_id` for isolated planning members.
  Not part of the live matrix.
- **Heterogeneous members.** Per-member `provider` / `model` / `temperature` /
  `reasoning_effort` (`TeamMemberSpec`), with `create_review_team` (PIPELINE preset,
  same-vendor warning) and `create_reflection_team` (rounds=3) presets. Unit-tested
  (17 tests); never run live, and never cross-vendor (the preset's actual purpose).
- **Member-granular durability** (FEP-0028 / ADR-023 / TD-25, shipped #733–#752):
  per-member checkpoint/resume at formation-natural granularity across all six,
  durable `MemberApprovalPause` for the four non-iterative formations, and
  `MemberEventSink` per-member streaming lanes. Not exercised over local inference.

### 1.3 Implemented but UNWIRED — the orphan trio

[`victor/coordination/formations/`](../../victor/coordination/formations/) contains three
`BaseFormationStrategy` subclasses with **zero references outside their own module** — not
in the enum, not in the coordinator's `_formations` dict, no presets, no docs:

| Strategy | What it does | State |
|---|---|---|
| `AdaptiveFormation` | switches formations mid-run on performance/error-rate/feedback thresholds (bounded by `max_switches`) | unit-tested; docstring references non-existent names (`OrchestrationFormation`, `HierarchyFormation`) — stale port |
| `DynamicRouterFormation` | analyzes the task and routes to ONE best-suited member (TaskAnalyzer category → keyword fallback → first agent) | unit-tested |
| `MultiLevelHierarchyFormation` | N-level coordinator→lead→member tree, divide-and-conquer with line/count/auto split strategies | unit-tested |

They are tested in `tests/unit/coordination/formations/test_new_formations.py`, so CI
keeps them green while nothing can reach them. This is the same drift class as the
InferFlux stale-copy finding (I3): code that looks supported but has no live path.

### 1.4 Designed, not implemented (or explicitly deferred)

- **FEP-0028 deferred remainder** (recorded as Non-Goals, not debt): iterative-formation
  pause/resume (CONSENSUS rounds, REFLECTION iterations cannot pause mid-loop),
  iterative mid-loop partial resume, member tool/token streaming granularity,
  `project.db` checkpointer, non-team chat continuation.
- **FEP-0006 external-harness executors** (Draft) — members backed by agents outside
  Victor's process; the remaining Omnigent cross-learning.
- **Not designed anywhere yet** (industry patterns with no Victor counterpart):
  group-chat / multi-party debate (AutoGen-style shared transcript with speaker
  selection), swarm/handoff routing (OpenAI Swarm-style agent-to-agent transfer),
  blackboard shared-memory coordination, market/auction task bidding,
  Mixture-of-Agents aggregation (proposer ensemble → aggregator),
  self-consistency ensembles (adjacent to FEP-0022 but formation-shaped).

## 2. Gaps (learnings-motivated, for the follow-up session)

Ordered by risk-to-correctness first. G-numbers are the handoff's work items.

- **G1 — CONSENSUS defaults defeat the formation.** `ConsensusFormation.__init__`
  defaults `max_rounds=1` (comment: "default: 1 for testing") and
  `agreement_threshold=0.7`; no preset overrides them, so the shipped default runs a
  single pass — there is no second round to reach consensus in. No tie-break policy
  exists; agreement comparison semantics on a 30B local model are untested.
- **G2 — REFLECTION verdict channel is fragile on small models.** Satisfaction is
  parsed from the critic's prose via `verdict\W+(satisfied|needs[ _]work)` with a
  keyword fallback. Small local models may omit the marker (loop then runs to
  `reflection_max_iterations` silently burning tokens) or emit it spuriously. A
  structured verdict (tool call or fenced marker) with a hard fallback policy is needed.
- **G3 — PARALLEL aggregation hides partial failure.** Any-member-success means a team
  reports success with 1/3 deliverables. Need: per-member outcome surfaced in
  `TeamResult` (exists as `MemberResult` but final_output synthesis ignores it), a
  partial-failure summary in `final_output`, and an optional per-member retry policy.
- **G4 — shared-path write races are documented, not fixed.** The live matrix required
  disjoint file paths per member (noted in the formations doc). Worktree isolation
  exists (§1.2) but is not default-on nor formation-aware for PARALLEL. Decide: opt-in
  per-member worktree for PARALLEL (config flag) vs. write-guard detection with a
  formation-aware error.
- **G5 — no capacity-aware admission control.** The R9700 serves 2 KV sequences; a
  3-member PARALLEL team silently serializes at the server. The formation never learns
  about provider capacity: `max_workers` exists but nothing derives it from the
  provider (InferFlux `max_parallel_sequences`), and no backpressure event reaches the
  coordinator (observability gap — members just run slower).
- **G6 — session-id isolation is pinned by unit tests, not by a live e2e assertion.**
  Direct spawns bind `resolve_member_session_id()` (dash format) on both `execute()` and
  the restored per-advance streaming binding. Unverified: nested spawns (member→grandchild
  propagation), and an e2e assertion that InferFlux actually sees DISTINCT
  `x-inferflux-session-id` values for concurrent members (needs server-side debug
  logging or a sandbox assertion).
- **G7 — heterogeneous presets never validated live.** The cost-optimal pattern (local
  workers + one cloud reviewer through Sandhi) has never been run. Also verify
  `reasoning_effort` stays stripped (capability-gated) for the InferFlux provider
  rather than erroring.
- **G8 — durability untested over local inference.** Pause/resume mid-PIPELINE with
  `MemberApprovalPause` against InferFlux; FEP-0028's deferred items stay deferred.
- **G9 — dynamic selection fails silently.** `formation_strategy` exceptions log at
  DEBUG and keep the default formation — indistinguishable from a strategy that
  intentionally returns the default. Needs a warning-level event on the teams→stream
  bridge. Live e2e of dynamic selection also missing.
- **G10 — orphan trio: integrate or delete.** Options: (a) wire DynamicRouter +
  MultiLevelHierarchy into the enum + `_formations` + presets (they implement the
  ITeamMember contract via the `_MemberContextAgent` shim); (b) archive behind a
  deprecation. `AdaptiveFormation` additionally needs its stale names fixed before any
  wiring. If deleting: `ARCHIVED_DOC_BANNERS` / `CANONICAL_POINTER_DOCS` hygiene
  applies, and `test_new_formations.py` coverage moves with them.
- **G11 — per-member cost attribution unverified end-to-end.** Sandhi stamps
  `x-sandhi-run-id` from the member session id; the InferFlux side keys session-KV on
  it. A member-tagged cost rollup (`GET /admin/usage/run/{run_id}` → per member) has
  never been reconciled against `TeamResult` metrics.
- **G12 — tool supply is not formation-aware.** Pruning is opt-in and off by default
  (correct for loop starvation), but PARALLEL with N members multiplies the full
  registry N times per turn across disjoint contexts. A formation-aware supply budget
  (e.g., intersect role `allowed_tools` before supply, which the live matrix did use)
  should be documented as the intended pattern rather than left implicit.
- **G13 — guard tuning is a two-model sample.** Narration/intent/refusal classifiers
  were tuned on Qwen3-Coder-30B + GLM-5.3 only. Before claiming edge-model support,
  run the same matrix on a small local model (qwen3.5:2b class) and record deltas.

## 3. Suggested follow-up session plan

Workstreams map to gaps; each is independently landable as PRs to develop.

1. **WS-A "Orphan trio decision" (G10)** — cheapest first: decide integrate-vs-archive,
   fix `AdaptiveFormation` stale docstrings either way, wire presets if integrating
   (enum values + `_formations` registration + `AgentTeam` preset + docs update).
2. **WS-B "Formation semantics hardening" (G1, G2, G3)** — consensus `max_rounds`
   default → 3 with a preset exposing it; tie-break policy (deterministic:
   supervisor/synthesis member breaks ties); REFLECTION structured verdict with
   hard-fallback; PARALLEL partial-failure synthesis in `final_output` + optional
   member retry. Table-driven unit tests per aggregation contract.
3. **WS-C "Capacity-aware parallelism" (G5, G12)** — derive default `max_workers` from
   provider-declared parallel capacity (InferFlux admin/metrics surface); emit a
   backpressure/throttle event on the `MemberEventSink`; document the formation-aware
   tool-supply pattern.
4. **WS-D "Isolation defaults" (G4, G6)** — PARALLEL opt-in per-member worktree;
   nested-spawn session-id propagation test; live e2e assertion of distinct
   `x-inferflux-session-id` per concurrent member.
5. **WS-E "Live validation sweep" (G7, G8, G9, G11)** — one session, live R9700:
   cross-vendor review preset (local workers + cloud reviewer), mid-PIPELINE
   pause/resume, dynamic `formation_strategy` e2e (+ warning event), member cost
   rollup reconciliation.
6. **WS-F "Edge-model guard battery" (G13)** — run the six-formation matrix on a small
   local model; file classifier deltas as follow-up fixes if they regress.

## 4. Pointers

- Live-matrix recipe and per-formation examples: [multiagent-formations-inferflux.md](multiagent-formations-inferflux.md)
- Durability contract: [FEP-0028](../../feps/fep-0028-team-node-durability-contract.md), ADR-023, TD-25 (roadmap)
- Formation strategies: [`victor/coordination/formations/`](../../victor/coordination/formations/); coordinator dispatch: `victor/teams/unified_coordinator.py` (`_formations`, `_execute_formation`)
- Session-id derivation: `SubAgentConfig.resolve_member_session_id` (`victor/agent/subagents/base.py`)
- Heterogeneous members: `victor/framework/teams.py` (`TeamMemberSpec`, presets)
- External-harness members: [FEP-0006](../../feps/fep-0006-external-harness-executors.md)
