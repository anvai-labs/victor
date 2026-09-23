# Multi-Agent Formation Coverage, Gaps, and Handoff

**Date:** 2026-09-17 · **Status:** Handoff for a follow-up session · **Live matrix:** R9700 / InferFlux / Qwen3-Coder-30B (see [multiagent-formations-inferflux.md](multiagent-formations-inferflux.md)) · **Rev 2:** adds §2, an industry pattern catalog researched from current framework docs and surveys (LangChain, Anthropic, AutoGen, OpenAI, CrewAI, ADK, MDPI/arXiv), with three new gaps (G14–G16) and two new workstreams (WS-G/WS-H). · **Rev 3:** adds §2.3 role-nomenclature standardization, §2.4 cross-cutting standards mandate (prompt engineering, design patterns, architecture), and workstream WS-I; adversarially reviewed — all codebase claims re-verified against the repo.

This document answers three questions: which formations Victor implements and which of
those were verified live; which designs exist beyond the canonical six (implemented,
unwired, or deferred); and what the follow-up session should address, with the
co-design learnings that motivate each item.

## 1. Coverage: what is implemented, and what was verified live

### 1.1 Canonical formations — implemented and live-tested (6/6)

`TeamFormation` ([`victor/teams/types.py`](https://github.com/anvai-labs/victor/blob/develop/victor/teams/types.py)) defines twelve values, all registered in `UnifiedTeamCoordinator._formations`.
The original six below were verified live on
2026-09-17 (single 3-member PARALLEL re-verified again on rebased develop, 48.2s green):

| Formation | Live result | Notes from the run |
|---|---|---|
| SEQUENTIAL | ✅ 30.0s | context chains member→member |
| PIPELINE | ✅ 18.0s | both stages delivered files |
| PARALLEL | ✅ 44.7s / 48.2s | 3 concurrent members, disjoint deliverables |
| HIERARCHICAL | ✅ 19.0s | supervisor delegation + synthesis |
| CONSENSUS | ✅ 18.5s | agreement reached in one pass |
| REFLECTION | ✅ 48.2s | generator→critic→refine, early exit |

The WS-F paired artifact battery (2026-09-18) uses a different strict task and
records failures separately: [matrix and follow-ups](multiagent-edge-model-battery.md).
Its Qwen 1/6 and small-MoE 0/6 results do not establish general edge-model support.

### 1.2 Additional surfaces — verification status

The original matrix omitted these surfaces; follow-up validation is recorded per item:

- **Dynamic formation selection.** `StateGraphNodeConfig.formation_strategy` picks a
  formation per invocation from graph state (sync-only; async strategies raise
  `TypeError`), and `formation_hint` / `topology_formation_hint` context keys override
  per call (`_resolve_effective_formation`, unified_coordinator.py). A raising strategy
  falls back to the default formation with a warning log and additive
  `team_formation_warning` stream/wire event (WS-E). Invalid returned identifiers
  use the same observable failure path.
- **Capacity admission** — ✅ WS-C ([PR #1110](https://github.com/anvai-labs/victor/pull/1110))
  verified three members against capacity two, with throttling and all assignments
  retained. Legacy `max_workers` alone still limits member count.
- **Worktree-isolated members.** `victor/teams/worktree_runtime.py`
  (`WorktreeAssignment`, `WorktreeExecutionPlan`) plus `worktree_planner`/`worktree_runtime`
  coordinator params and colon-format `child_session_id` for isolated planning members.
  ✅ WS-D live validation covered materialized per-member worktrees, independent
  pytest, and wire/result session attribution ([PR #1111](https://github.com/anvai-labs/victor/pull/1111)).
- **Heterogeneous members.** Per-member `provider` / `model` / `temperature` /
  `reasoning_effort` (`TeamMemberSpec`), with `create_review_team` (PIPELINE preset,
  same-vendor warning) and `create_reflection_team` (rounds=3) presets. Unit-tested
  (heterogeneous-member and review-preset suites). ✅ WS-E ([PR #1115](https://github.com/anvai-labs/victor/pull/1115)) verified six
  R9700/Qwen members and a ZAI/GLM reviewer through Sandhi, including per-member
  model routing and local reasoning-effort stripping.
- **Member-granular durability** (FEP-0028 / ADR-023 / TD-25, shipped #733–#752):
  per-member checkpoint/resume at formation-natural granularity across all six,
  durable `MemberApprovalPause` for the four non-iterative formations, and
  `MemberEventSink` per-member streaming lanes. ✅ WS-E ([PR #1115](https://github.com/anvai-labs/victor/pull/1115)) verified an injected
  mid-pipeline approval pause and resume on the mixed local/cloud team; completed
  writer and reviewer executions were restored without replay.

### 1.3 Additional formations — WS-A integration

**Decision: INTEGRATE.** The orphan trio now has enum values, shared-registry
registration, `AgentTeam` presets, feature/formation docs, coordinator-dispatch tests,
and explicit `supports_durable_pause() == False` statements. The new ZAI/Sandhi
matrix passed adaptive and dynamic router; multi-level hierarchy failed missing
member artifacts. Matched Qwen validation remains outstanding (see §1.5).

| Strategy | Public formation / preset | State |
|---|---|---|
| `AdaptiveFormation` | ADAPTIVE / `create_adaptive_team` | ✅ Integrated; invocation-local bounded switching, real member outcomes; [PR #1107](https://github.com/anvai-labs/victor/pull/1107) |
| `DynamicRouterFormation` | DYNAMIC_ROUTER / `create_router_team` | ✅ Integrated; one-member dispatch retaining structured outcomes; [PR #1107](https://github.com/anvai-labs/victor/pull/1107) |
| `MultiLevelHierarchyFormation` | MULTI_LEVEL_HIERARCHY / `create_multi_level_hierarchy_team` | ✅ Integrated; validated member-ID tree, lossless splitting, supervisor synthesis; [PR #1107](https://github.com/anvai-labs/victor/pull/1107) |

Regression coverage: `tests/unit/coordination/formations/test_new_formations.py`
and the mirrored `test_adaptive.py`, `test_dynamic_router.py`,
`test_multi_level_hierarchy.py`, and `test___init__.py`, plus preset coverage in
`tests/unit/teams/test_integrated_formations.py`. Durable partial resume is not
supported for these three: no topology/routing/tree cursor is persisted. Approval
remains inline (see G17).

### 1.4 Designed, not implemented (or explicitly deferred)

- **FEP-0028 deferred remainder** (recorded as Non-Goals, not debt): iterative-formation
  pause/resume (CONSENSUS rounds, REFLECTION iterations cannot pause mid-loop),
  iterative mid-loop partial resume, member tool/token streaming granularity,
  `project.db` checkpointer, non-team chat continuation.
- **FEP-0006 external-harness executors** (Draft) — members backed by agents outside
  Victor's process; the remaining Omnigent cross-learning.
- **Patterns with no Victor counterpart**: see the researched catalog in §2 — the
  blackboard and contract-net remain absent. Conversation-native formations now
  use the FEP-0035 transcript substrate; ensemble status is tracked separately (G16).

### 1.5 Completion audit (updated 2026-09-23; cohort dates retained)

Implementation delivery and live acceptance have different denominators:

| Measure | Complete | Remaining |
|---|---|---|
| WS-A through WS-I increments landed | 9/9 (100%); PRs in §4 | 0 original increments |
| Historical live passes before the new matrix | 7/15 (47%): original six plus ensemble vote | Historical tasks/gates differ; not matched acceptance |
| 2026-09-20 contract-v1 ZAI/Sandhi case passes | 10/15 (67%) | 2 deliverable failures; 3 ensemble cases blocked by resource exhaustion |
| Historical contract-v1 matched Qwen/InferFlux case passes | 0/15 (0%); not run under that contract | 15 cases |
| Historical contract-v1 matched matrix case passes | 10/30 (33%) | 20 cases without a pass (67%); overall ZAI run remains FAIL |
| 2026-09-21 renewed contract-v2 ZAI/Sandhi acceptance | 15/15 (100%); overall PASS | No case failures; broader lifecycle and cross-model acceptance remain separate |
| Contract-v2 Qwen/InferFlux ROCm case checks | 0/15 (0%); overall FAIL | 15 cases; missing tests and formation-contract failures |
| Contract-v2 combined case checks | 15/30 (50%) | 15 Qwen cases without a pass; not mixed-team or lifecycle acceptance |
| Historical 2026-09-21 single-file Qwen/ROCm profile | 12/15 (80%); overall FAIL | Debate, multi-level hierarchy and ensemble synthesizer |
| Historical 2026-09-21 single-file ZAI profile | 0/4 completed cases; interrupted/incomplete | 26 HTTP 429s among 61 calls; no profile acceptance |
| 2026-09-23 OIDC ZAI standard cohort | 15/15 (100%); overall PASS | Broader lifecycle acceptance remains separate |
| 2026-09-23 OIDC ZAI single-file cohort | 15/15 (100%); overall PASS | No case failures; separate task profile |
| 2026-09-23 OIDC Qwen3/ROCm single-file cohort | 0/15 accepted; interrupted overall FAIL | First completed case timed out; second cancelled; 13 unstarted, not 15 model-quality failures |
| OIDC Qwen14/CUDA single-file cohort | 0/15; not started | Held for consolidated-origin liveness investigation |
| Current six-Qwen/one-ZAI C5 acceptance | Open; historical failed run retained | Full corrected run held for origin liveness, then reviewed verdict on InferFlux #184 |

The new OIDC cohorts use clean Victor source `208e2535f523b3c28a77823ff90673c329970a5f`
([#1173](https://github.com/anvai-labs/victor/pull/1173)), released Sandhi 0.9.0 and
separate inference/accounting identities. The
[standard cohort](evidence/zai-oidc-standard-2026-09-23.json) passed all 15 cases
with 120 HTTP-200 calls and 120 clean wire/SQLite/C4 joins. The
[single-file cohort](evidence/zai-oidc-single-2026-09-23.json) separately passed all
15 with 121 calls/joins. Each has 33 distinct member sessions, 29 passing pytest
checks and 29 independent numeric oracles of 10,242 inputs each. Explicit cache
reporting is 120/120 and 121/121; dashboard totals and attribution reconcile.
Only the standard cohort's **accounting** identity renewed; inference-token renewal
was not exercised. These are new actual-member experiments, not historical replays.

The [new Qwen3 attempt](evidence/qwen3-oidc-single-2026-09-23.json),
`matrix-96be2e7e91`, is incomplete and **FAIL**. Its first sequential case hit the
240-second case limit after a gateway HTTP 504 (126.015 seconds observed including
credential acquisition). The parallel case was cancelled while in flight. Four
observed gateway HTTP calls across three sessions were captured; zero clean accounting
joins passed, and cancellation-time accounting failed. The final wire file records
four HTTP 504s; its completion occurred after the accounting snapshot, which
contains only one ledger row. Preserve both timings and the failed verdict;
later ledger writes cannot retroactively make that snapshot pass.
The runtime reported ready while generation completion counters stayed zero and
its queue grew. This is an origin-liveness investigation (G52), not evidence that
Qwen cannot solve the simpler task. Qwen14 and the new C5 run remain unstarted.

[Security and runtime evidence](evidence/oidc-consolidated-runtime-2026-09-23.json)
records released Sandhi [v0.9.0](https://github.com/anvai-labs/sandhi/releases/tag/v0.9.0),
source `d755c262386e6cd041a300532ea34411d012a2d8`, independently verified GitHub,
PyPI, npm and crates artifacts. Both Homebrew binaries report 0.9.0 after a real
upgrade ([tap #66](https://github.com/anvai-labs/homebrew-tap/pull/66)); release
history is synchronized to develop ([Sandhi #290](https://github.com/anvai-labs/sandhi/pull/290)).
Eight live authorization checks passed, including denied member admin/dashboard
access and denied accounting mutation/inference. TLS-verified browser sign-in,
actual-member run lookup, matching C4/SQLite totals and logout passed. The earlier
[0.8.0 57-token check](evidence/sandhi-080-oidc-release-2026-09-23.json) remains
preserved separately. Security/release checks alone add no formation passes.

Consolidated InferFlux source `02cf22addb9f78309bb5b2807427f215353a7085` serves all
three pinned models on loopback 8080: Qwen3 on ROCm, Qwen14 and BGE on CUDA. CPU CI,
GPU CI and the separate same-process dual-GPU gate passed before deployment. Fresh
model-file hashes and placement are recorded in the runtime evidence. Those gates
used short setup workloads; they do not establish Victor acceptance. The origin
still uses existing private API keys. Direct-origin OIDC, embedding compatibility
through the new gateway, streaming/cancellation, tokenizer equivalence, session
leases and executed-cache acceptance remain separate open gates.

The [renewed contract-v2 ZAI run](evidence/zai-formation-matrix-v2-2026-09-21.json),
`matrix-aae4713703`, used clean source `d872b6103e319fa562d20a514d458af52314c027`
(the candidate merged through [PR #1163](https://github.com/anvai-labs/victor/pull/1163)).
All 12 formations and three ensemble modes passed: 119 HTTP-200 calls, 33 distinct
member sessions, 62 required artifacts, 29 passing pytest checks and 29 independent
numeric oracles covering 10,242 inputs each. All 119 wire/SQLite/C4 joins and the
dashboard totals/attribution reconciled: 80,548 fresh input, 363,520 cache-read and
19,013 output tokens, with explicit cache reporting for 119/119 calls. Maximum HTTP
latency was 8.936 seconds; maximum ledger settlement was 0.001888 seconds. This is
new evidence, not a replacement verdict for either failed earlier run. It proves
these ZAI buffered cases, not C5 mixed-team, origin cancellation, executed cache
reuse, tokenizer equivalence or full resource lifecycle acceptance.

The [new Qwen ROCm baseline](evidence/qwen-rocm-formation-matrix-v2-2026-09-21.json),
`matrix-94d9707bc9`, used the same clean Victor source and numeric contract. It failed
all 15 cases, predominantly missing member-written test files, with additional
conversation/ensemble response-contract failures. All 130 wire/SQLite/C4 joins and
aggregate dashboard conservation passed across 29 observed sessions: 457,640 fresh
input, 12,867 output and 44 reported cache-read tokens. This new ROCm cohort does not
replace C5's preserved CUDA runtime. Next evaluate a simpler, explicitly labelled
single-file task profile while retaining pytest, independent numeric oracles,
formation decisions, distinct sessions and full accounting. Neither the failed
baseline nor a future simpler pass establishes a model-size explanation.

The opt-in single-file profile landed in [PR #1165](https://github.com/anvai-labs/victor/pull/1165).
The [new Qwen run](evidence/qwen-rocm-single-file-2026-09-21.json),
`matrix-98a78c934d`, passed 12/15 cases with all 125 HTTP-200 calls and 33 sessions
reconciled (449,330 fresh input, 11,373 output, 12 reported cache-read tokens).
Debate's judge never wrote its decision while reporting that pytest was unavailable;
multi-level hierarchy omitted the first member's file; the synthesizer's saved
answer differed from its final output. Those strict failures remain failures.
The [ZAI reference attempt](evidence/zai-single-file-2026-09-21.json),
`matrix-5920dd66b7`, was interrupted after repeated rate limits: 26 HTTP 429s and
35 HTTP 200s, four completed failed cases and a cancelled in-flight case. Its
cancellation-time reconciliation also failed; some wire records finalized later
than that snapshot, so those HTTP-200 responses are not clean-join evidence. This
neither closes single-file ZAI acceptance nor invalidates the earlier standard
15/15 pass. Both used clean source `09f9894072c0c9dbe67cbb5e768e86a21a17f9fa`.
Do not combine single-file and standard task counts. A model-size explanation is
still unproven; preserve tool-follow-through and output-contract failures separately
from transport, accounting and capacity observations.

The [new actual-member ZAI experiment](evidence/zai-formation-matrix-2026-09-20.json)
passed sequential, parallel, hierarchical, pipeline, consensus, group chat, debate,
handoff, adaptive and dynamic router. Reflection lacked `review.json`; multi-level
hierarchy lacked `second.py`, `test_second.py`, `third.py` and `test_third.py`.
All three ensemble cases failed during file-descriptor exhaustion (G41), so they
do not measure model quality. The original overall verdict remains **FAIL**.
Separate read-only reconciliation matched all 83 calls across 25 sessions against
wire, SQLite, C4 and dashboard, with no further model calls or unrelated ledger
rows. It does not replace the failed verdict or establish lifecycle acceptance.

Before the renewed run, the union of historical and passing cases was 12/15 (80%), spanning different
models, tasks and gates: it is coverage evidence, not a matched comparison.
Historical WS-F strict acceptance remains Qwen 1/6 and LFM 0/6. No weighted overall
completion percentage is asserted, and case percentages do not estimate effort.
The remaining correctness work includes G32 completion, G34 buffered reporting
and G41 resource exhaustion. G39's native member task binding is implemented in
[PR #1161](https://github.com/anvai-labs/victor/pull/1161); live acceptance remains
separate. G43 member-pytest cleanup landed in
[PR #1160](https://github.com/anvai-labs/victor/pull/1160) after all CI gates passed;
the fix adds no live passes.
G36 structured verification landed in [PR #1159](https://github.com/anvai-labs/victor/pull/1159)
after all CI gates passed, including Vertical Py3.12. G42 independent numeric oracles landed in
[PR #1157](https://github.com/anvai-labs/victor/pull/1157) with all CI green, including
Vertical Py3.12. The new contract-v2 experiment `matrix-7cce287098` ran on clean
source `f3792270d72db06c12093c2992ff1817c2e79859`. All 12 formation cases passed
their local gates, including reflection and multi-level hierarchy; none of the
three ensemble modes passed. Across 80 HTTP-200 calls and 25 member sessions,
75 wire/SQLite joins passed and five failed, with an additional ledger call-count
failure. The overall verdict remains FAIL. File descriptors grew from 9 before
the first case to 253 after the ensemble failures; G41 remains open. These are
new actual-member results, not a replay or a retrospective change to v1 evidence.
Do not combine v1/v2 pass counts. The [research evaluation](multiagent-formation-research-evaluation.md)
maps relevant papers to existing code, evidence and ordered follow-ups; it adds no
live passes. G41's scoped cache-owner correction landed in
[PR #1155](https://github.com/anvai-labs/victor/pull/1155), with full CI including
Vertical Py3.12 green; complete matrix resource acceptance remains outstanding.
G40 reflection result retention landed in [PR #1152](https://github.com/anvai-labs/victor/pull/1152)
after all required CI, including Vertical Py3.12, passed. Both reflection members'
sessions and usage reconciled in the new run; its missing artifact still fails.
G31's cross-repository cache/lifecycle investigation remains open; G17/G21 durability
limitations remain explicitly deferred. A model-size diagnosis is not established
by these failures: ZAI also returned successful members with missing deliverables.
The 2026-09-21 read-only preflight restored the Mac loopback tunnel on 18081,
confirmed Qwen ready through the configured Sandhi route, and matched the preserved
gateway and origin binary hashes. The accepted origin is still partial CUDA,
not a new R9700/ROCm deployment. Readiness checks establish no new model-call or
formation acceptance. G45 separately tracks a test-collection MLX crash.

### 1.6 Surface and test pruning audit (2026-09-21)

Keep the 12 canonical formations and three aggregation policies. They already
share one registry, the conversation substrate, and `execute_ensemble`; aggregation
policies are not additional formations. Sequential continues after member failure,
whereas pipeline stops. Consensus iterates agreement; ensemble vote uses one
independent proposal wave. Debate judges a transcript; ensemble judge selects an
independent candidate. A synthesizer combines proposals instead of selecting one.
Router selects a member; adaptive switches registered strategies. Recursive
multi-level hierarchy differs from supervisor delegation. Removing these surfaces
would remove supported behavior rather than duplicate execution code.

Pruned three older adaptive tests whose metadata/switch-bound assertions are
covered more strongly by existing dispatch tests, and one unused nested writer
helper. All executed production line and branch sets were unchanged in the paired
coverage run (136 tests before, 133 after). Preserve compatibility, registry,
preset and dispatch tests because they exercise different boundaries. Identical
member/coordinator test helpers remain a possible mechanical consolidation; no
assertions were removed on that basis. No production formation was removed.

## 2. Industry pattern catalog (researched 2026-09-17)

Sources fetched for this section: LangChain "Choosing the right multi-agent
architecture", Anthropic "Building effective agents", AutoGen agent-chat team guides
(SelectorGroupChat, Swarm), OpenAI Agents SDK handoffs, CrewAI processes, Google ADK
multi-agent docs, and two 2026 surveys (MDPI Future Internet orchestration survey;
arXiv 2601.13671). URLs in §5.

Two orthogonal dimensions organize every pattern found: **control topology** (who
directs whom) and **communication substrate** (what carries information between
members). Victor's six formations span topology well but sit on exactly ONE
substrate — which is the deepest structural finding of this review.

### 2.1 Control-topology patterns

| Pattern (a.k.a.) | Mechanism | Victor status |
|---|---|---|
| Sequential chain (prompt chaining, CrewAI sequential, ADK `SequentialAgent`) | fixed order; each stage consumes prior output | ✅ SEQUENTIAL / PIPELINE |
| Fan-out / fan-in (parallelization-sectioning, ADK `ParallelAgent`) | independent subtasks concurrently, aggregate after | ✅ PARALLEL (aggregation gaps: G3) |
| Router (dispatch-and-synthesize; LangChain "router", Anthropic "routing") | classify input → invoke one/few specialists → synthesize | Integrated (`DynamicRouterFormation`, §1.3; [PR #1107](https://github.com/anvai-labs/victor/pull/1107)) |
| Supervisor / orchestrator-workers (LangChain subagents, CrewAI hierarchical with `manager_llm`) | central agent decomposes, delegates, synthesizes; workers stateless to each other | ✅ HIERARCHICAL (single level) |
| Hierarchical multi-level (ADK transfer trees, org-chart topologies) | coordinator → leads → members, aggregate up | Integrated (`MultiLevelHierarchyFormation`, §1.3; [PR #1107](https://github.com/anvai-labs/victor/pull/1107)) |
| Group chat with speaker selection (AutoGen `SelectorGroupChat`, `RoundRobinGroupChat`) | members broadcast to a SHARED TRANSCRIPT; LLM/selector/round-robin picks next speaker; termination conditions | ✅ GROUP_CHAT; bounded shared transcript, router/callback/round-robin selection (WS-G) |
| Swarm / peer handoff (OpenAI Agents SDK handoffs, AutoGen Swarm) | control MOVES agent-to-agent via handoff-as-tool-call; receiving agent continues with carried context | ✅ HANDOFF; structured peer destination with carried transcript (WS-G) |
| Evaluator-optimizer (generator-critic loop) | generate → critique → refine until satisfied | ✅ REFLECTION (verdict fragility: G2) |
| Adaptive / dynamic topology switching (MDPI "adaptivity" dimension; Magentic-One replanning) | monitor progress → switch topology or replan mid-run | Integrated (`AdaptiveFormation`, §1.3; [PR #1107](https://github.com/anvai-labs/victor/pull/1107)); Magentic-style ledger replanning not designed |
| Ensemble aggregation (self-consistency, "More Agents Is All You Need" voting, Mixture-of-Agents layered aggregation) | N proposals of the SAME task → vote / layered aggregation | ✅ opt-in ensemble vote/judge/synthesizer; R9700 voting validated (WS-H) |
| Structured debate (Du et al. multiagent debate) | adversarial rounds with a judge; improves factuality | ✅ DEBATE; bounded contributions followed by one judge (WS-G) |
| Blackboard shared memory (Hearsay-II lineage) | specialists watch/mutate a shared workspace opportunistically | ❌ absent — Victor's `shared_state` dict is coordinator-curated, not opportunistic |
| Contract-net / auction task bidding (Smith 1980) | manager announces tasks; agents bid on capability/load | ❌ absent |
| External/inter-system agents (A2A protocol, MCP ecosystems, FEP-0006) | members are remote/foreign agents behind a protocol | 🚧 FEP-0006 Draft; Sandhi already gives the transport seam |

### 2.2 Communication substrates (the missing dimension)

| Substrate | Who uses it | Victor today |
|---|---|---|
| Artifact/context handoff (task string in → result out) | LangChain subagents-as-tools, OpenAI `Agent.as_tool()`, ADK `AgentTool` | ✅ the ONLY substrate: members get the task + `shared_state`, return `MemberResult` |
| Shared transcript (broadcast conversation) | AutoGen group chats, OpenAI handoffs (full history carries), debate | ❌ none — each member owns a private orchestrator history |
| Shared state/scratchpad keyed per member | ADK session state (`output_key`), blackboard | ⚠️ `TeamContext`/`shared_state` dict exists but is coordinator-curated, not member-writable by convention |
| Structured ledgers (task/progress) | Magentic-One TaskLedger/ProgressLedger | ❌ (adjacent: session ledger FEP-0023, not team-facing) |

**Structural conclusion:** every Victor formation is a *workflow shape over artifact
handoff*. That is the same design point as Anthropic's orchestrator-workers and
LangChain's subagents — the empirically strongest patterns for coding work — so the
six formations are NOT behind on the patterns that matter most for Victor's domain.
The gaps are concentrated in the *conversation-native* family (group chat, debate,
swarm handoff), which requires building a transcript substrate before any of those
three topologies can exist; and in *ensemble aggregation*, which is substrate-light
and could ride PARALLEL.

### 2.3 Agent-role nomenclature (standardization proposal)

Role names are inconsistent across the ecosystem — and inside Victor. Sources for
this subsection: Anthropic's multi-agent research system (LeadResearcher / subagents /
CitationAgent), Magentic-One (Orchestrator + WebSurfer/FileSurfer/Coder/
ComputerTerminal), CrewAI (`manager_llm`/`manager_agent`), AutoGen (`UserProxyAgent`,
planning-agent example), OpenAI Agents SDK (triage/handoff examples), A2A (client
agent / remote agent, donated to the Linux Foundation), and the 2026 surveys
(worker/specialist/service/support role families).

**Victor's terms today (from code):**

| Concept | Victor name(s) | Inconsistency |
|---|---|---|
| Control-plane agent | `TeamAgentCategory.SUPERVISOR`; `_active_supervisor()` | WS-I: `explicit_supervisor_id` emitted; deprecated manager input alias normalized with warning ([PR #1109](https://github.com/anvai-labs/victor/pull/1109)) |
| Team execution unit | `TeamMember` / "member" | "worker" (`max_workers`), "sub-agent" (docs), `_MemberContextAgent` |
| Spawned child agent | `SubAgent` / `SubAgentConfig` | consistent |
| Domain roles | `SubAgentRole` (researcher, planner, executor, reviewer, …) | consistent; but `formation_role` adds generator/critic strings for REFLECTION |
| Iterative evaluator | "critic" (REFLECTION formation_role) | "reviewer" (role) vs "critic" (formation role) vs "evaluator" (Anthropic's pattern name) used interchangeably in prose |

**Ecosystem synonym map (what external material calls each concept):**

| Victor canonical | Ecosystem synonyms | Source examples |
|---|---|---|
| **supervisor** (control plane) | orchestrator, manager, lead agent, coordinator, triage | Magentic-One "Orchestrator"; CrewAI `manager_agent`; Anthropic "LeadResearcher"/"lead agent"; Kore.ai/Medium "coordinator"; OpenAI "triage" |
| **member** (execution unit) | worker, specialist, subagent, surfers/coders (domain-named) | "supervisor-worker" literature; Magentic-One specialists; Anthropic "subagents" |
| **subagent** (spawned child) | subagent, sub-agent, child agent | LangChain/Anthropic |
| **reviewer** (single-pass review) | reviewer, inspector, checker | Victor's review preset |
| **critic** (iterative-loop evaluator) | critic, evaluator | REFLECTION; Anthropic evaluator-optimizer |
| **judge** (one-shot verdict over candidates) | judge, adjudicator, LLM-as-judge | debate/MoA literature — needed by WS-G/WS-H |
| **synthesizer** (composes final output) | synthesizer, aggregator, summarizer | Anthropic lead "synthesizes"; MoA "aggregation layer" |
| **router** (classifies → dispatches) | router, dispatcher, triage | LangChain router; OpenAI triage |
| **approval gate** (HITL stop) | user proxy, human-in-the-loop agent | AutoGen `UserProxyAgent` ↔ Victor `MemberApprovalPause` |
| **client/remote member** (external, FEP-0006) | client agent, remote agent (A2A) | A2A spec (Linux Foundation) |

**Mandate for the follow-up session:**

1. One canonical term per concept, per the table above. Docs lead with the canonical
   term and mention synonyms once; new APIs accept canonical names only.
2. Collapse the supervisor/manager dual keys (`explicit_supervisor_id` +
   `explicit_manager_id`) to one key; keep the other as a deprecated alias.
3. Do NOT use "planner" for the control plane — it is a `SubAgentRole` domain role;
   AutoGen's planning-agent example conflates the two and copying that usage would
   break Victor's role/category split.
4. Reserve **critic** for iterative-loop evaluators (REFLECTION), **judge** for
   one-shot verdicts over candidate outputs (WS-H ensembles, WS-G debate), and
   **reviewer** for single-pass review — the WS-G/WS-H formation APIs must use these
   names in their `formation_role` strings.
5. Use **member** when the formation-agnostic execution unit is meant; keep
   `max_workers` as a compat API name but do not add new "worker" surfaces.
6. For FEP-0006 external members, adopt A2A's client/remote terminology verbatim.

### 2.4 Cross-cutting standards mandate (prompting, patterns, architecture)

The follow-up session's workstreams must follow these, drawn from the researched
sources and Victor's own codesign lessons:

**Prompt engineering**
- Member task prompts must state: objective, output format, tool/source guidance,
  and explicit boundaries — Anthropic's research system found vague delegation
  ("research the semiconductor shortage") caused duplicated work across subagents.
- Prefer structured, machine-checkable output contracts over prose parsing: the
  REFLECTION `VERDICT:` line is the right shape; generalize it (or move to
  tool-call-shaped verdicts where the provider supports tools — G2), never add new
  parse-the-prose contracts.
- Keep selector/speaker/aggregation prompts minimal for small local models — AutoGen
  explicitly warns that condition-heavy selector prompts break on smaller models and
  to move complexity into a programmatic selector function instead (maps to G13).
- Encode per-delegation scale guidance in supervisor/delegation prompts where
  Anthropic does ("1 subagent with 3–10 tool calls for fact-finding; 10+ subagents
  for complex research" is prompt text in their system): prompts control how much
  work one member gets; member-COUNT adaptation is code-side and belongs to WS-C.
- Few-shot anchor one example per output contract for edge models (G13).

**Software design patterns**
- One registry, one dispatch: formations stay strategy classes behind the
  `_formations` dict (no parallel dispatch chains — the #353 split-brain lesson).
- One derivation per identifier (session ids, formation roles, event names) — the
  codesign consolidation rule; a second copy is a defect even when it matches today.
- Additive, opt-in contracts with byte-identical defaults (FEP-0028's "absent
  checkpointer ⇒ byte-identical" is the model; apply to WS-G/WS-H surfaces).
- New event variants need a consumer-decision row before landing — the rule is
  recorded in [foundations-strategy-2026-07.md](foundations-strategy-2026-07.md)
  (referred to elsewhere as the TD-0008 rule); applies to WS-C backpressure events
  and WS-G speaker/handoff events.
- No silent fallbacks: every degrade path emits a warning-level event (G9 is the
  counterexample being fixed, not the pattern).
- Dead code is either wired or archived with banner + canonical pointer — never left
  green-and-unreachable (G10; the doc-cleanup hygiene registries apply).

**Architectural principles**
- Subagents are compression boundaries: members return condensed findings or file
  references, not raw tool dumps (Anthropic's "game of telephone" and
  pass-references-not-payloads findings); pairs with G12's supply budget.
- Capacity-aware defaults over static limits (WS-C): member counts and concurrency
  derive from provider capacity and task class.
- Isolation at trust/capacity boundaries: session-id per member, worktree per member
  where writes may collide (WS-D).
- Guard/classifier changes land through the FEP-0025 paired-gate process with a
  recorded experiment — measure before tuning (G13).
- Definition-of-done for ANY new formation: strategy module + enum value +
  `_formations` registration + preset + docs (features.md + formations doc) +
  coordinator-dispatch tests + durability statement (`supports_durable_pause()`).
  G10 exists because the orphan trio has only the strategy module and (non-dispatch)
  unit tests — five of the seven are missing.

### 2.5 What this research changes in the plan

- The orphan trio maps cleanly onto validated mainstream patterns (router,
  multi-level hierarchy, adaptivity) — strengthening the INTEGRATE option in G10:
  these are not exotic leftovers, they are the three topologies every major framework
  ships, and all three are unit-tested in-repo.
- Ensemble aggregation (G16) is new work but substrate-light: N-sample-one-task can
  reuse PARALLEL execution + a vote/aggregate step; debate/group-chat need G14 first.
- FEP-worthiness: a shared-transcript + peer-handoff substrate is a public-surface
  change (new formation family, new communication contract) → draft an FEP rather
  than landing it as a formation strategy (per the TD-0008 consumer-decision rule).

## 3. Gaps (learnings-motivated, for the follow-up session)

Ordered by risk-to-correctness first. G-numbers are the handoff's work items.
G1–G13 come from the co-design sessions and code audit; G14–G16 from the §2 research.

- **G1 — ✅ consensus semantics hardened (WS-B, [PR #1108](https://github.com/anvai-labs/victor/pull/1108)).** Default rounds are 3;
  agreement compares explicit consensus keys or exact output values rather than
  successful execution. The consensus preset exposes rounds/threshold and optional
  supervisor tie-break; unresolved disagreement fails the team. Live small-model
  comparison validation remains part of the later sweep.
- **G2 — ✅ structured reflection contract added (WS-B, [PR #1108](https://github.com/anvai-labs/victor/pull/1108)).** Opt-in JSON
  verdicts have strict validation and immediate failure on malformed output. The
  legacy default remains byte-compatible and warns on keyword fallback; use the JSON
  contract for new local-model runs. Checkpoint resume preserves the verdict format.
- **G3 — ✅ partial failure surfaced (WS-B, [PR #1108](https://github.com/anvai-labs/victor/pull/1108)).** Parallel synthesis includes
  failed-member summaries; opt-in per-member retries retain attempt costs, stop at
  approval pauses, and complete before durable member checkpointing.
- **G4 — ✅ opt-in PARALLEL worktrees (WS-D, [PR #1111](https://github.com/anvai-labs/victor/pull/1111)).** `parallel_worktree_isolation`
  materializes one worktree per member, forwards the assigned directory through
  public spawn, and binds supported file/shell tools without process-wide `chdir`.
  Missing worktrees or tool adapters fail explicitly. Worktrees are preserved for
  review by default. The live test wrote identical relative filenames in all three
  worktrees, with no parent-directory writes and three passing independent tests.
- **G5 — ✅ opt-in capacity-aware admission (WS-C, [PR #1110](https://github.com/anvai-labs/victor/pull/1110)).** `capacity_aware_parallelism`
  queries the provider declaration and bounds simultaneous members without dropping
  assignments. Saturated members emit `member_throttled` through the sink and v1
  stream bridge. R9700 live validation passed with three members at capacity two;
  automatic ROCm discovery remains an explicit upstream limitation (G18).
- **G6 — ✅ nested and live session isolation verified (WS-D, [PR #1111](https://github.com/anvai-labs/victor/pull/1111)).** Nested member
  execution inherits the immediate parent's session and restores its caller.
  Coordinator dispatch forwards configured member identities; observed live
  `x-inferflux-session-id` headers match all three configured member IDs and the
  per-member result metadata. Evidence is linked in the InferFlux formation recipe.
- **G7 — ✅ cross-vendor live validation (WS-E, [PR #1115](https://github.com/anvai-labs/victor/pull/1115)).**
  R9700 Qwen3-Coder-30B handled six members (24 calls); GLM-5.3 reviewed the local
  writer's artifact (9 calls, 8,083 reasoning tokens). The local payloads omit
  reasoning_effort. All traffic traversed Sandhi with seven distinct member sessions.
- **G8 — ✅ local/cloud durability validation (WS-E, [PR #1115](https://github.com/anvai-labs/victor/pull/1115)).**
  An injected MemberApprovalPause before the reviser preserved public pause fields;
  resume restored the writer/reviewer and executed only the reviser. The live test
  uses MemoryCheckpointer in-process; FEP-0028's iterative non-goals remain.
- **G9 — ✅ observable dynamic selection (WS-E, [PR #1115](https://github.com/anvai-labs/victor/pull/1115)).** Exceptions and invalid formation
  identifiers emit warning logs plus `team_formation_warning` through the member
  sink, client stream, and v1 wire. The ZAI harness asserts PARALLEL selection and
  warned default dispatch. Async selectors retain explicit TypeError rejection.
- **G10 — ✅ orphan trio: INTEGRATE (WS-A, [PR #1107](https://github.com/anvai-labs/victor/pull/1107)).** All three have public enum,
  registry, preset, docs, dispatch-test, and durability surfaces. Adaptive stale
  names and duplicate dispatch were removed. Regression tests cover real member
  execution/failure, concurrent adaptive calls, lossless task splitting, and invalid
  trees. Original six defaults remain unchanged.
- **G11 — ✅ member accounting reconciliation (WS-E, [PR #1115](https://github.com/anvai-labs/victor/pull/1115)).**
  Opt-in capture_member_usage exposes neutral counters in metadata; retries and
  recovery responses retain every attempt. Seven mixed-provider members reconciled
  exact input/output/total counts with Sandhi, including ZAI cache/reasoning counts.
  Fourteen Python files plus review.json and seven independent pytest tests passed
  in 308.15s. [Mixed-run evidence](evidence/ws-e-mixed-gateway.json) records IDs/routes.
- **G12 — ✅ formation-aware tool supply documented (WS-C, [PR #1110](https://github.com/anvai-labs/victor/pull/1110)).** Member `allowed_tools`
  narrows the registry before provider supply. The live capacity run supplied four
  filesystem/shell tools per member; global pruning defaults remain unchanged.
- **G13 — ✅ edge-model battery executed (WS-F, [PR #1118](https://github.com/anvai-labs/victor/pull/1118)).** The same
  six-formation harness ran against Qwen3-Coder-30B/ROCm and the available
  LFM2.5-8B-A1B/CPU small MoE through Sandhi. Strict artifact gates passed 1/6 and
  0/6 respectively; failures and deadline limits are retained, not reported as
  support. [Matrix and evidence](multiagent-edge-model-battery.md) record the
  differences and follow-ups G32/G33. No classifiers were tuned; dense 2B support
  remains unvalidated.
- **G14 — ✅ shared-transcript substrate and conversation formations (WS-G, [PR #1113](https://github.com/anvai-labs/victor/pull/1113)).**
  FEP-0035 ([PR #1112](https://github.com/anvai-labs/victor/pull/1112)) defines bounded immutable transcript snapshots, explicit speaker
  selection/termination, and consumer decisions. GROUP_CHAT and DEBATE have enum,
  registry, preset, docs, dispatch-test, and explicit non-durable contracts.
- **G15 — ✅ peer-to-peer control transfer (WS-G, [PR #1113](https://github.com/anvai-labs/victor/pull/1113)).** HANDOFF consumes validated
  peer destinations and carries the shared transcript; invalid/self destinations
  fail explicitly, and cycles are bounded by max_turns. Typed `PeerHandoff`
  records cross the member sink and v1 wire bridge.
- **G16 — ✅ ensemble aggregation (WS-H, [PR #1114](https://github.com/anvai-labs/victor/pull/1114)).** One shared policy implements independent
  proposals followed by strict-majority vote, one-shot judge, or a synthesizer pass.
  Public `create_ensemble_team` and CONSENSUS `mode="vote"` presets use validated
  JSON contracts. R9700 voting passed with three deliverables/test pairs and
  distinct member wire sessions. Judge/synthesizer modes have dispatch tests.

WS-I status: ✅ canonical `FormationRole` identifiers and supervisor-key normalization
implemented ([PR #1109](https://github.com/anvai-labs/victor/pull/1109)). Review and reflection presets bind reviewer/critic roles
without mutating caller specs. Compatibility manager methods and `max_workers` remain.

- **G17 — integrated trio has no durable partial resume.** WS-A makes this explicit:
  adaptive requires a topology/attempt cursor; hierarchy requires a recursive cursor;
  router requires persisted selection. `supports_durable_pause()` is false, and
  member approvals stay inline. A separate durability design/test increment is
  required before claiming member-granular resume for these formations.

- **G18 — historical ROCm sequence-capacity discovery gap; current contract drift is G51.** Verified live during
  WS-C: `/v1/admin/models` omits `max_parallel_sequences`; `/metrics` exposes only
  CUDA capacity gauges (zero on this ROCm backend). The active serving YAML declares
  2 sequences and the process environment sets `INFERFLUX_LLAMA_CTX_SIZE=65536`.
  WS-C accepts that verified operator declaration on the provider, and fails clearly
  if admission is requested without a declaration. Automatic ROCm discovery requires
  an upstream InferFlux admin/metrics addition; no model-window heuristic is used.

- **G19 — ✅ public spawn identity/context forwarding (WS-D, [PR #1111](https://github.com/anvai-labs/victor/pull/1111)).**
  The coordinator forwards configured member/session and worktree identity to spawn.
  Compact result attribution retains the resolved wire session ID without copying
  full response payloads. The live worktree run matches actual headers to all three
  configured member IDs and result metadata.
- **G20 — generated test completion can overstate validation.** During the WS-C
  Qwen3-Coder-30B run, members wrote module-level assertions and reported success
  after pytest exited 5 (no collected tests). The external validation harness
  correctly rejected the run. Delegation now anchors an explicit `def test_*`
  contract; general completion-classifier changes require the WS-F paired-gate
  experiment process rather than an unmeasured heuristic patch.

- **G21 — conversation and ensemble durable replay is not implemented.** New
  conversation modes reject checkpoint/resume before execution. Pending speaker
  selection, transcript restoration, and in-flight peer transfers need a dedicated
  durable-state design. This is explicit; no partial replay is claimed.

- **G22 — ✅ public pipeline pause metadata loss (WS-E, [PR #1115](https://github.com/anvai-labs/victor/pull/1115)).** The spawn adapter dropped
  `awaiting_approval` and `approval_request`, and TeamResult dropped aggregate pause
  fields. Both boundaries now preserve the structured signal. Absent pause state,
  TeamResult serialization remains unchanged.
- **G23 — ✅ usage writes mutated snapshots (WS-E, [PR #1115](https://github.com/anvai-labs/victor/pull/1115)).** SessionStateAccessor returned
  a copy while runtime writers and metrics expected a live accumulator. Internal
  access now preserves one dictionary's identity, including assignment; the public
  SessionStateManager snapshot remains defensive. Real-runtime regression tests
  cover inclusive input, output, cache, reasoning, and accumulator assignment visibility.
- **G24 — ✅ explicit member overrides bypassed gateways (WS-E, [PR #1115](https://github.com/anvai-labs/victor/pull/1115)).** Overrides now use
  the existing canonical gateway resolver for configured provider blocks/environment,
  pass the gateway to the managed factory, and use its virtual key. Invalid explicit
  gateway setup fails closed. Legacy direct-provider warned inheritance remains.
- **G25 — Sandhi transparent ZAI requests omitted JSON Content-Type.** Fixed and
  live-tested in [Sandhi PR #265](https://github.com/anvai-labs/sandhi/pull/265);
  23 raw-forwarding tests passed and all CI gates are green. Repository policy
  requires an approving review before merge; the patched local binary is in use.

- **G26 — ✅ lazy grammar availability in codegraph CI ([PR #1117](https://github.com/anvai-labs/victor/pull/1117)).**
  tree-sitter-language-pack 1.20 downloads grammars lazily. CI now preloads every
  required grammar with bounded retries, failing explicitly instead of silently
  skipping language tests. Both Python matrix jobs pass all 106 codegraph tests.
- **G27 — ✅ recovery responses omitted from member usage (WS-E, [PR #1115](https://github.com/anvai-labs/victor/pull/1115)).**
  ResponseCompleter now returns structured provider responses; the existing runtime
  accumulator counts each once, including empty retry responses and error recovery.
  A rejected first mixed run exposed the discrepancy; real-completer regression
  tests cover both recovery paths. The successful rerun reconciled every member.
- **G28 — task completion can outlive missing deliverables.** In the first mixed
  run, a local member repeated a successful write, exhausted its loop, and returned
  a successful recovery summary without the requested test file. The independent
  artifact gate rejected it. A clearer task passed the rerun; classifiers were not
  changed. Follow-up must use the paired guard experiment process before tuning.
- **G29 — ✅ task-report API tokens use the shared session accumulator ([PR #1121](https://github.com/anvai-labs/victor/pull/1121)).**
  Buffered members and recovery calls updated the live accumulator while task
  reports preferred the streaming-only cost tracker, producing zero API tokens.
  Reports now derive prompt/completion/total deltas from the same counters used by
  member usage. Regression coverage includes task boundaries, failed/empty tasks,
  empty recovery attempts, and real streaming finalization with mixed reasoning
  conventions; 115 related tests passed. Remaining tracker fields are G34.
- **G30 — ✅ shared usage survives reset and checkpoint restore ([PR #1120](https://github.com/anvai-labs/victor/pull/1120)).**
  SessionStateManager retains the live accumulator when replacing execution state,
  so runtime writers and metrics readers remain attached after either reset mode
  and checkpoint restore. Restored values are copied to avoid aliasing caller-owned
  checkpoint data. A regression reproduces the original stale-reference failure
  and verifies subsequent response accounting through all three transitions;
  124 session, metrics, runtime, and size-guard tests passed.

- **G31 — cache observability needs a paired workload replay.** The mixed Victor run
  reported zero InferFlux cache tokens, but controlled direct/gateway probes showed
  426 cached tokens for repeated chat and 514 with tools. Sandhi stored both correctly.
  JSON/logprob paths bypass reuse in the serving InferFlux build; this is not proven
  to explain the original member workload. Owner handoffs were written on aiserver1
  at /home/vsingh/code/sandhi/docs/upstream/inferflux-cache-codesign-2026-09-18.md and
  /home/vsingh/code/inferflux/docs/planning/SANDHI_CACHE_CODESIGN_HANDOFF_2026-09-18.md.
  Follow-ups cover actual-reuse accounting, path diagnostics, and distinguishing
  absent cache fields from explicit zero. No caching heuristic was changed in Victor.

- **G32 — edge/baseline completion guard gaps (WS-F, [PR #1118](https://github.com/anvai-labs/victor/pull/1118)).** Successful member status
  did not imply complete artifacts. Qwen omitted requested test files in four
  formations and emitted invalid REFLECTION JSON; LFM emitted proposed commands
  as final text and literal backslash-n in Python files. Four LFM cases also hit
  240s deadlines with gateway 504s, which are not proven classifier defects.
  Follow-up: retain both model traces, isolate tool/prose and artifact-validity
  decisions under FEP-0025, and separately measure CPU admission/deadline behavior.
  Keep strict structured verdicts and independent pytest as acceptance gates.
  A subsequent [mixed gateway replay](evidence/g29-mixed-gateway.json) on G29's
  accounting fix reproduced missing `test_writer.py` and `test_fallbackb.py`.
  Five collected tests passed, but the full run correctly failed the required
  artifact gate. Its seven task reports matched member counters and gateway totals
  across 29 calls with seven distinct member sessions. This accounting regression
  run is not a paired prompt experiment and does not establish model improvement.
  The [2026-09-19 C5 mixed attempt](evidence/c5-mixed-gateway-2026-09-19.json)
  again failed artifact completeness: `fallbacka.py` and `test_fallbacka.py` were
  absent. Six delivered tests passed; seven distinct member sessions made 27
  successful calls. Wire/SQLite/C4/dashboard conservation passed, but this does
  not close G32 or the full mixed-team gate. See G37/G38 for additional limits.
- **G33 — ✅ deterministic spin-guard regression coverage ([PR #1119](https://github.com/anvai-labs/victor/pull/1119)).** The
  failing test mixed classifier/plugin initialization with its loop deadline and
  could pass the iteration assertion after an unrelated early failure. Tests now
  supply deterministic TaskAnalysis while retaining the real perception/offload
  seam, loop counters, and timeout limits. A nonterminal workload must execute
  exactly 1 or 4 iterations; perception must run once per iteration. The original
  20s assertion and 30s timeout remain. Five focused tests passed in 2.66s, and
  135 completion/intent/perception tests passed. Production classifiers are unchanged.
- **G34 — buffered task-report cache/cost/request fields lack tracker integration.**
  G29 fixes API token deltas, but task-report cache read/write, monetary cost, and
  request count still use SessionCostTracker, which only streaming finalization
  updates. Buffered member usage/cache evidence remains separately available.
  Follow-up must record each buffered provider response (including recovery) once
  through the metrics owner, preserve per-call reasoning conventions, and reconcile
  mixed buffered/streaming task boundaries without adding duplicated token totals.
  Zero tracker values for buffered tasks are not evidence of zero cost or requests.
- **G35 — ✅ verification retry exhaustion cannot accept COMPLETE ([PR #1122](https://github.com/anvai-labs/victor/pull/1122)).**
  The shared buffered/streaming verification gate previously skipped zero-budget
  checks and the last repair attempt, and could leave failed verification marked
  COMPLETE when the outer iteration limit fired. It now checks every completion
  claim, records structured results, and changes unsuccessful evaluations to RETRY
  or FAIL with zero reward score. Configured verification bypasses cached prose;
  unsupported StateGraph and iteration-stream paths fail explicitly. Default runs
  without a verifier retain their behavior. Validation: 81 focused gate/loop tests
  plus 127 loop/integration/session-ledger/size-guard tests passed. This establishes
  prerequisite correctness for G32 experiments, not evidence that either model now
  passes the formation battery.
- **G36 — ✅ structured built-in verifier acceptance ([PR #1159](https://github.com/anvai-labs/victor/pull/1159)).**
  LocalTestVerifier parses test-count prose and can ignore a nonzero process exit
  if parsed passed/total counts match. LintVerifier derives success from colon-bearing
  output lines rather than the exit status. Follow-up must use explicit process
  success plus runner-owned structured reports, cover empty/failed/timeout runs,
  and retain diagnostics without promoting model-written prose into evidence.
  **Implementation:** LocalTestVerifier now requires a fresh pytest JUnit report,
  validates its tree and outcome totals, excludes skipped-only/empty runs, and
  counts process success as an additional explicit check. Nonzero exit, missing,
  malformed or contradictory reports cannot pass. Other test runners require a
  custom Verifier until their structured adapters exist; failed detection does not
  substitute pytest. LintVerifier uses one process-status check instead of inferred
  punctuation counts. Buffered subprocess execution retains diagnostic tails,
  separately bounds cleanup, avoids inherited pipes and preserves cancellation
  with cleanup notes. Already-exited processes are never signalled. These changes
  affect configured built-in verifiers only; default runs remain unchanged. This
  does not repair G43's separate live-harness pytest path or establish test quality
  or live completion. The new direct suite replaces no existing tests: the audit
  found no prior LocalTestVerifier/LintVerifier suite or duplicate coverage owner.
- **G37 — ✅ failure evidence retention ([PR #1143](https://github.com/anvai-labs/victor/pull/1143)).**
  In the [C5 mixed attempt](evidence/c5-mixed-gateway-2026-09-19.json), the
  unchanged `multiagent_gateway_live.py` asserted the review verdict before its
  canonical member-usage reconciliation, independent pytest, and final
  `evidence.json` write. Its `finally` block retained only request metadata.
  An external observer preserved 27 wire/SQLite/C4 joins and dashboard deltas,
  and an independent pytest invocation checked delivered files; neither substitutes
  for the unexecuted canonical member-usage check. Follow-up must retain structured
  partial team/member results and all applicable checks before returning a failed
  verdict, including exception/timeout paths, without relaxing any assertion.
  **Implementation:** the harness now collects review, per-member usage, artifact
  presence and bounded independent pytest results before saving a structured failed
  verdict. Execution failures and cancellation retain returned partial pipeline results;
  one rejected review or failed usage lookup cannot suppress other checks. Private
  reports include the task-contract version and restore the caller's working
  directory and gateway environment after execution. This fixes evidence retention,
  not G32 member completion or C5 acceptance. Cancellation during final checks,
  unavailable Git provenance, and unsupported/cyclic member metadata also produce
  failed reports without preventing process-state cleanup. In-flight outcomes not
  returned by the runtime are not recovered. HTTP errors/timeouts followed by
  successful provider retries still require the external gateway observer; this
  change does not certify those paths.
- **G38 — ✅ explicit numeric task contract v2 ([PR #1143](https://github.com/anvai-labs/victor/pull/1143)).**
  The same run's ZAI reviewer returned structured `needs_work` for `writer(x) = x * 2`:
  overflowing floats and sequence inputs violate the requested addition invariant.
  Both counterexamples were independently reproduced from the delivered function.
  The harness nevertheless requires `approved`; the reviewer verdict and failed
  run remain intact. Follow-up must define and review the task's numeric domain
  and expected review outcome before any new experiment. Do not relabel this
  verdict or repeat calls merely to obtain an approved verdict.
  **New experiment contract v2:** writer, reviewer and all other members receive
  the same explicit domain: Python ints/floats that are integer multiples of 0.25
  in [-1024, 1024]; x, y and x+y must remain in that domain for the invariant.
  Booleans, strings/sequences, non-finite values and overflow are out of scope.
  These bounded binary fractions preserve exact arithmetic for the required
  operations. A correct implementation should receive a structured `approved`
  review; `needs_work`, malformed findings, missing files or failed pytest still
  fail acceptance. This is new evidence when run, not a reinterpretation of v1.

- **G39 — ✅ structured native member task binding ([PR #1161](https://github.com/anvai-labs/victor/pull/1161)).**
  The coordinator's SubAgent adapter suppresses the static member goal whenever
  the dispatched task differs from the team goal, treating that difference as
  dynamic delegation. Conversation and other structured formation tasks also
  differ, so a member can receive its response contract without its declared
  artifact assignment. An offline GROUP_CHAT reproduction returned success while
  the captured SubAgent task omitted the unique member assignment. This is an
  offline task-binding finding, not evidence of model quality or a live pass.
  The opt-in gateway matrix explicitly carries assignments keyed by the preset's
  canonical member IDs inside its structured team objective. Existing coordinator
  defaults remain unchanged.
  **Implementation:** `member_task_binding="structured-v1"` opts the native
  SubAgent adapter into one structured input envelope binding the configured member
  ID, name, role and assignment beside the unchanged formation task. The formation
  task owns active delegation and its response contract; it outranks the static
  assignment. No task-text comparison decides whether to include identity or
  assignment in this mode. Unknown modes fail before spawning; absent/null values
  preserve previous task bytes. The option uses run-local context and the existing
  canonical member IDs, and all matrix cases opt in. The conversation dispatch
  regression failed before repair; the hierarchy regression now checks each actual
  member's bound identity/assignment instead of merely searching all assignments
  in the shared goal. No new formation, provider call or live pass is established.
- **G40 — ✅ reflection per-member results and accounting retained ([PR #1152](https://github.com/anvai-labs/victor/pull/1152)).**
  An offline public-preset/coordinator reproduction supplied generator and critic
  responses with distinct sessions and usage, but the successful TeamResult
  contained only the synthetic `reflection_formation` aggregate. The context-agent
  adapter reduces each response to text; the strategy returns that aggregate
  without either member's metadata. Strict matrix session/usage acceptance cannot
  pass this shape, even when both members deliver valid artifacts. Preserve the
  assertions; fix result retention in a separate compatible runtime increment
  before spending calls on new live reflection acceptance. This finding does not
  invalidate the differently scoped historical reflection experiment.
  **Implementation:** with the existing `capture_member_usage=True` opt-in,
  reflection executes canonical participants and retains each generator/critic
  result, session, accumulated counters and attempt history. Partial and terminal
  iteration checkpoints retain the same evidence without replaying completed
  iterations. A failed member, malformed verdict, missing usage or changed session
  fails explicitly. The default aggregate result and legacy checkpoint payload stay
  unchanged. Coordinator final output remains the generated solution rather than
  concatenating critique text. This corrects result retention; live acceptance
  still requires the matrix's artifact, pytest and independent accounting gates.

- **G41 — multi-case runtime exhausts file descriptors.** The new actual-member
  ZAI matrix reached 12 formations and 83 successful calls, then encountered
  `OSError: [Errno 24] Too many open files` during ensemble setup/execution.
  All three ensemble cases failed; SQLite/provenance access and final accounting
  also failed. The original failed evidence is preserved. A separate read-only
  reconciliation verified the original ledger boundary and all 83 calls without
  new inference, but does not close this lifecycle defect. Diagnose resource
  ownership with bounded offline measurements, add a failing lifecycle regression,
  and fix the measured owner without closing borrowed providers/shared services.
  Do not attribute these ensemble failures to model quality or hide them by
  increasing descriptor limits. Root cause is not yet established.
  **Scoped lifecycle correction — ✅ [PR #1155](https://github.com/anvai-labs/victor/pull/1155):** an offline, no-inference probe isolated one
  reproducible contributor: five parent/member pairs retained ten tool-cache
  SQLite connections after parent shutdown and garbage collection. Explicit
  owned-cache cleanup returns to the nine-descriptor baseline after every pair,
  with no retained cache connections. Parent shutdown and terminal buffered/
  streaming spawn cleanup now close their own handles through LifecycleManager;
  borrowed providers and shared services are not shut down for member cleanup.
  Persisted cache entries survive closure. Close errors remain explicit and
  retryable; cancellation and primary failures retain their identity and cleanup
  diagnostics. This does not prove that cache handles explain the entire live
  failure. G41 stays open pending renewed multi-case measurement and acceptance.
  **Further measured owner — ✅ [PR #1163](https://github.com/anvai-labs/victor/pull/1163):** background conversation persistence
  opens thread-local project SQLite connections on executor threads that outlive
  members. An offline five-pair execution retained 15 project connections after
  shutdown and garbage collection. `ChatService` now releases only the calling
  worker's connection after each persistence job through `ConversationStore`;
  committed data and other threads' handles remain available. Real SQLite tests
  cover successful/failed writes and independent caller handles; cleanup failures
  are observable and cannot replace a propagating cancellation. This is a scoped
  repair, not a full-matrix acceptance claim.
  **Provider-owned typed handles — ✅ [PR #1164](https://github.com/anvai-labs/victor/pull/1164):** an offline real-binding probe
  against a local mock HTTP server retained idle Rust transport pools after native
  provider close: five closed providers grew descriptors from 8 to 21. Releasing
  only each closing provider's cached typed providers/runtime held the same probe
  flat at 11 after initialization. The mixin now releases these owned references
  in `finally`, preserving native close errors and cancellation identity. Repeated
  close and another live provider's ownership are covered; no member cleanup closes
  a borrowed parent provider. This separate repair was not present in the passing
  ZAI run above. The run removes the ensemble exhaustion blocker for this cohort,
  but does not establish every provider lifecycle boundary.
  TDD covered success, failure, timeout, cancellation, early stream closure,
  persistence, and cleanup-error paths. The duplicate bootstrapper presence-only
  test was removed: the stronger lazy-proxy test preserves the same 51 executed
  source lines and branch coverage. Cache-clear unit tests now use temporary
  directories instead of the developer's persistent cache.

- **G42 — ✅ independent matrix numeric oracles ([PR #1157](https://github.com/anvai-labs/victor/pull/1157)).** The opt-in matrix's
  contract-v1 ordinary and ensemble doubling tasks request a member-authored test of `f(4) == 8`;
  independent pytest reruns that same test. A new isolated offline counterexample,
  `def first(x): return 8`, passes the requested pytest while returning 8 rather
  than 10 for input 5. This probes the test predicate only, with zero inference;
  it is not an actual-member replay or a full matrix acceptance result. Preserve
  historical verdicts under their recorded gates. Define a bounded task domain
  before the next experiment, freeze runner-owned independent oracles, and use
  TDD to reject known wrong implementations, missing/modified reports, nonzero
  process exits and timeouts. Member-authored tests remain required deliverables.
  The [research audit](multiagent-formation-research-evaluation.md) records the
  counterexample and relevance of ExecCritic/SWE-Bench Pro Verified without claiming
  their training algorithms or benchmarks are reproduced in Victor.
  **Implementation (contract v2):** the opt-in matrix now checks every implementation
  against a frozen runner-owned oracle over 2,049 integer inputs and 8,193 quarter-step
  float inputs in [-1024, 1024]. It requires complete structured counts, a zero
  process exit and unchanged oracle/artifact hashes, independently of member-authored
  pytest. Missing, malformed, oversized or partial reports, wrong numeric behavior,
  changed files and cleanup failures cannot pass. The subprocess receives no gateway
  credentials or pytest plugins. Execution and cleanup have separate deadlines;
  cancellation remains primary, and file output prevents detached descendants from
  retaining pipes and extending the wait. Detached child containment is not claimed.
  The initial constant-return false pass and both adversarial cleanup findings were
  reproduced before repair. Existing scenario/acceptance suites were extended;
  their ownership audit found no redundant tests to remove. All prior artifact,
  session, accounting and deadline checks remain required. No new live acceptance
  is established, and G36's built-in framework verifiers remain separate work.

- **G43 — ✅ bounded member-pytest cleanup ([PR #1160](https://github.com/anvai-labs/victor/pull/1160)).** The matrix's
  existing member-authored pytest subprocess still captures output through a pipe
  and awaits an unbounded `process.wait()` after killing pytest on timeout. A separate
  disposable offline reproduction of that block, with only its deadline reduced
  from 60 seconds to one second, returned after 3.610 seconds when a detached child
  inherited the outer pipe. Pytest exited -9; the child was self-limited and cleaned
  up. A control using pytest's normal capture returned after 1.003 seconds. This
  did not invoke the corrected numeric oracle and is not actual-member evidence.
  Repair this separate ownership/deadline path with TDD before the next live matrix:
  bound execution and cleanup, preserve cancellation and diagnostics, and avoid
  borrowed/shared process termination. Do not merely increase the timeout.
  **Implementation:** the actual `check_case` regression reproduced a 5.151-second
  return for a one-second deadline before repair. The member-pytest path now uses
  the built-in verifier's buffered process helper: file-backed output, the unchanged
  60-second execution deadline and a separate one-second cleanup grace. Evidence
  distinguishes the actual child exit from runner timeout/cleanup status, retains
  diagnostic tails (4,000 bytes per stream) with explicit truncation flags, and
  records cleanup faults. A zero child exit cannot override failed cleanup.
  Cancellation propagates; it is not converted into a normal failed subprocess.
  The verifier tuple contract and unconfigured agent defaults remain unchanged.
  Test ownership remains split between helper lifecycle tests and actual harness
  integration; the audit found no redundant cases to remove. This offline repair
  establishes neither detached-child containment nor new live formation acceptance.

- **G44 — ✅ isolated matrix scenario test imports ([PR #1161](https://github.com/anvai-labs/victor/pull/1161)).**
  The changed-file CI selection imported the scenario test before the gateway test;
  console pytest did not inject the repository root, so the scenario's sibling
  `scripts.validation` import failed. Local `python -m pytest` had masked that
  dependency. A standalone safe-path collection reproduced the exact missing-module
  error. The scenario test now loads repository scripts explicitly, independently
  of another test's import side effects. The same five CI-selected suites are
  exercised with the implicit current-directory import removed. Production
  behavior and live evidence are unchanged; no duplicate test case was added.

- **G45 — ✅ isolated MLX unit-test collection ([PR #1162](https://github.com/anvai-labs/victor/pull/1162)).** During the
  2026-09-21 collection check, a Python child aborted in MLX 0.30.6's Metal device
  initialization with an empty-array exception. `test_mlx_provider.py` evaluated a
  runtime-availability subprocess in a module-level skip marker, even when pytest
  was only collecting tests. Collection completed, but the child generated a macOS
  crash report. This matches the upstream [empty-device report](https://github.com/ml-explore/mlx/issues/3148);
  it is not InferFlux server or gateway execution evidence.
  **Implementation:** remove the collection-time probe and use a fake MLX backend
  in adapter unit tests, retaining their assertions and deterministic loader-error
  coverage. A guarded import reproduction rejects both subprocess probing and
  direct native imports without invoking Metal. Existing registry laziness tests
  remain separate: registry loading and test-module collection are different
  boundaries. No redundant tests were found. The four previously skipped hardware
  integration cases remain explicitly skipped; this fixes test isolation, not the
  installed MLX native runtime, and establishes no MLX or formation live acceptance.

- **G46 — ✅ bounded ledger observation ([PR #1163](https://github.com/anvai-labs/victor/pull/1163)).**
  New actual-member run `matrix-7cce287098` on 2026-09-21 recorded 80 HTTP-200
  responses and 25 distinct member sessions. Its original reconciliation passed
  75 joins, rejected ordinals 0, 3, 16, 22 and 43 as `nonunique_request_join`,
  and failed `wire_ledger_call_count`. Each rejected request recorded identical
  before/after SQLite row bounds and no origin request ID. This does not establish
  whether rows were absent, delayed, or excluded by the correlation window.
  Preserve the failed evidence; investigate the existing ledger read-only and
  compare explicit session/step identities before changing any join logic.
  A supplementary reconciliation must not overwrite the original FAIL, and no
  correlation or conservation assertion may be weakened. The private original is
  `/private/tmp/victor-zai-matrix-v2-20260921/matrix-7cce287098/evidence.json`.
  Read-only inspection found all five exact session/step rows immediately above
  the recorded response-time row bounds. The observer now allows up to one second
  for ledger settlement, retaining the original response bound and recording its
  wait and matched/ambiguous/timeout status. The final join still requires exactly
  one row within the observed bounds and the same session/run/step/model identity;
  HTTP attempts, 120-second acceptance deadline, C4, cache and token conservation
  checks remain enforced. Missing/duplicate rows fail explicitly. This change
  makes no retrospective change to the original failed verdict. Renewed run
  `matrix-aae4713703` passed all 119 exact joins and aggregate conservation checks.

- **G47 — consensus final-round usage omits earlier rounds.** New actual-member
  Qwen ROCm run `matrix-94d9707bc9` exhausted three consensus rounds. Aggregate
  wire/SQLite/C4/dashboard accounting passed, but both member usage checks failed:
  the first member reported 6,725 input / 262 output against session totals
  44,830 / 1,200; the second reported 41,104 inclusive input / 902 output against
  76,233 / 1,681. `ConsensusFormation` returns the final round's results while
  the same canonical member sessions cover all rounds. Follow-up must retain
  per-attempt evidence and aggregate canonical usage under the existing opt-in
  capture contract, preserving legacy defaults and round-boundary durability.
  Missing/inconsistent session or usage data must fail explicitly. The simpler
  task profile changes neither this accounting defect nor its acceptance gate.
  The scoped repair reuses the reflection attempt reducer for consensus capture:
  sum all retained deltas and tool/duration totals, preserve failed attempts and
  reject inconsistent sessions/counters. Capture snapshots on intake and restore
  prevent mutable member metadata or returned aggregates from corrupting history.
  Round-boundary and terminal resume retain totals without replay; changing capture
  mode during resume is rejected. Legacy capture-disabled output is unchanged.
  TDD reproduced five accounting cases plus two independent-review alias defects;
  existing durability tests were parameterized instead of duplicated. This repair
  adds no live multi-round consensus acceptance by itself.

- **G48 — repeated conversation speaker usage is last-turn-only (code audit).**
  ✅ Code repair landed in [#1169](https://github.com/anvai-labs/victor/pull/1169)
  after all CI gates passed; repeated-speaker live acceptance remains separate.
  `ConversationFormation.record` sums tool/duration counts across repeated speakers
  but replaces usage with the latest turn's metadata. Current two-speaker matrix
  cases execute each speaker once and do not test this boundary. Reuse the shared
  attempt reducer under the existing opt-in capture contract and add a repeated-
  speaker coordinator test before claiming complete conversation accounting.
  The scoped repair now snapshots captured results before consuming each turn and
  uses the existing attempt reducer for speaking members, routers and judges.
  Captured usage, tool calls and duration cover the whole member session; malformed
  final turns retain their failed attempt, and inconsistent session/usage fails
  explicitly. Capture-disabled behavior remains unchanged. Public coordinator
  regressions cover all three conversation formations and repeated router calls,
  including shared mutable provider metadata. Existing protocol/selection tests
  retain ownership; no durable partial resume or new live acceptance is claimed.

- **G49 — SSO migration: ZAI actual-member acceptance passed; local/mixed acceptance open.** Sandhi's
  default OIDC, dashboard roles and scoped machine authorization landed in
  [Sandhi #281](https://github.com/anvai-labs/sandhi/pull/281); consistent 0.8.0
  source versions landed in [#282](https://github.com/anvai-labs/sandhi/pull/282).
  These source changes do not establish published-binary or Victor acceptance.
  New evidence from source candidate `d6dc91bbd6f1c748f0ee41d220467a1e385d8522`
  used a 900-second Kanidm machine token for one successful ZAI request and
  correctly denied admin access. Run `oidc-machine-1790121988166631691` reconciled
  23 input / 20 output / zero cache / 43 billable tokens across wire, SQLite and
  the TLS-verified browser dashboard, with explicit cache coverage 1/1. This was
  a machine integration check, not an actual Victor member or C5 run.

  Published-binary verification now passed separately: release source
  `7b634c8653c434a4f11afd9dd6916fd8bdae3aff`, binary SHA-256
  `ee678487322c96487d51d8ccd19ab4fe0c3a4c08094c6c1280f57c414e82a083`, and new run
  `oidc-machine-1790137208264702760` are recorded in the
  [release evidence](evidence/sandhi-080-oidc-release-2026-09-23.json). That run
  conserves 57 tokens; it does not replace the earlier 43-token source-candidate
  evidence or add formation acceptance. A gateway request ID was absent from the
  wire response: C4/SQLite request-ID and run/session/step joins were checked
  separately, without inventing an equality to the upstream request ID.

  InferFlux strict token claims landed in
  [#212](https://github.com/anvai-labs/inferflux/pull/212), and buffered/streaming
  TLS peer identity verification landed in
  [#213](https://github.com/anvai-labs/inferflux/pull/213). The audit credential
  exposure fix landed in [#214](https://github.com/anvai-labs/inferflux/pull/214).
  Promotion [#216](https://github.com/anvai-labs/inferflux/pull/216) merged these
  repairs into main at `4c71bdc998580bab75bdbd5bb71972ed9a914019`. Deployment was initially
  held: that CUDA runtime gate failed before its model-backed step
  ([run 35818698581](https://github.com/anvai-labs/inferflux/actions/runs/35818698581)).
  That failure is preserved; [#217](https://github.com/anvai-labs/inferflux/pull/217)
  repaired the missing TLS-probe build dependency, promoted through
  [#218](https://github.com/anvai-labs/inferflux/pull/218). CPU, CUDA/ROCm and
  same-process dual-GPU gates passed on `02cf22addb9f78309bb5b2807427f215353a7085`;
  the exact accepted binary is deployed. Actual-member liveness still fails (G52).
  Verified issuer discovery, Kanidm access-token interoperability, explicit
  authorization policy and bounded authority requests
  still need acceptance. Do not infer these from signed-token unit tests or the
  Sandhi browser sign-in result.

  Keep gateway routing independent from authentication: direct origin access is
  an explicit, authenticated route and bypasses gateway metering/budgets; an auth
  failure must never switch routes or weaken credentials. Browser sessions,
  machine credentials and upstream provider credentials have separate roles and
  audiences. OIDC remains the default SSO posture; token compatibility requires
  explicit selection. Preserve existing credentials, historical evidence and cache.

  ✅ [#1173](https://github.com/anvai-labs/victor/pull/1173) landed after all CI
  gates passed, including Vertical Py3.12. Both live validation harnesses expose an additive
  [OAuth broker profile](gateway-validation-oauth.md), with one bounded credential
  adapter, fixed routes and separate accounting identity. Legacy invocations still
  read virtual-key and admin-token files. The fixed Kanidm broker and separate
  accounting service account are provisioned; both new ZAI cohorts passed all
  15 cases with the released OIDC gateway (§1.5). Local/mixed and inference-token
  renewal acceptance remain open. Sandhi 0.8.0
  deliberately requires `admin` for the read-only C4 diagnostics POST (ADR-0011);
  a `viewer` cannot satisfy unchanged C4 assertions. A separately reviewed scoped
  diagnostics permission merged through
  [Sandhi #285](https://github.com/anvai-labs/sandhi/pull/285); it is a post-0.8.0
  increment published and deployed in 0.9.0; live accounting uses viewer plus that
  permission and is denied inference, config and budget mutation. Do not silently
  grant admin or omit diagnostics. Keep renewable OAuth access tokens inside the
  observer adapter rather than member environments or cached provider handles;
  use fixed route/grant bindings and separate inference/accounting identities.
  Bootstrap credentials stay with their trusted broker. Retain
  all existing request/session, usage, dashboard, deliverable and pytest assertions;
  a 120-second buffered gateway timeout remains an acceptance failure. Released-binary
  SSO and both ZAI references passed. Next resolve origin liveness, validate the
  simpler Qwen ROCm/CUDA cohorts and inference-credential renewal, then run the
  six-Qwen/one-ZAI C5 gate. Measure performance after security and usability;
  no formation row turns green from the SSO integration check alone.

- **G50 — subagent retry can conceal an authentication denial (code audit).**
  ✅ Code repair landed in [#1170](https://github.com/anvai-labs/victor/pull/1170)
  after all CI gates passed, including Vertical Py3.12.
  The outer `SubAgent` retry catches `ProviderError`, including the canonical
  `ProviderAuthError` which the provider layer already excludes from retry.
  A denied request can therefore re-enter the whole chat and repeat earlier tool
  work. The scoped correction propagates that typed error immediately and keeps
  the existing failed-member result. Existing retry tests retain transient cooldown
  coverage; new 401/403 regressions require one chat invocation, no backoff, and a
  failed member even when a later response would succeed. TDD reproduced both
  failures. This does not establish OAuth renewal, upstream identity, or live C5
  acceptance; those remain G49. The redundant enum-count test was removed because
  the existing exact role-set assertion already iterates and checks all five roles.

- **G51 — current InferFlux capacity metadata does not match Victor's lookup.**
  The deployed admin model response publishes `runtime.sequence_capacity` and
  `runtime.context_tokens_per_sequence`. Victor's provider still requests a flat
  `max_parallel_sequences` under the configured base URL's `/admin/models` suffix;
  a `/v1` base also changes that path. Qwen14's static 32,768-token policy exceeds
  the currently served 16,384 tokens per sequence. Validate one authoritative
  runtime capability contract and its authenticated direct/gateway access before
  claiming capacity-aware acceptance. Do not change global model metadata to match
  one deployment or silently treat missing capacity as unlimited.

- **G52 — consolidated-origin readiness did not establish actual-member liveness.**
  The new OIDC Qwen3 cohort failed at the unchanged 120-second gateway bound and
  240-second case limit (§1.5). Readiness remained green while observed generation
  completion counters were zero and queued work accumulated. Runtime-wide counters
  can include other clients; they are not per-member proof. The
  [read-only 11:03 UTC snapshot](evidence/inferflux-stall-observation-2026-09-23.json)
  still reports ready, a queue depth of eight and zero model completion counters
  after this cohort stopped. Encoder-context log
  messages are not evidence of Qwen-to-BGE misrouting without request correlation.
  Investigate accepted-source dispatch, shared embedding/generation scheduling and
  cancellation before more matrix/C5 load. Preserve shared cache, failed evidence,
  runtime identity and the prior passing setup gate; do not replace this failure
  with a longer-timeout result or attribute it to model quality.

- **G53 — standalone gateway deadline policy is not operator-configurable per route.**
  Sandhi 0.9.0 has shared transport timeout primitives, but standalone provider
  construction supplies no timeout overrides: buffered completion is 120 seconds,
  stream setup 30 seconds, stream idle 90 seconds. The requested policy should
  resolve exact model-within-endpoint → endpoint credential reference → global
  defaults, with an operator-owned ceiling and validated positive finite values.
  Resolve each field once after authentication/model authorization, expose the
  effective value and source to authorized operators, and apply the same result to
  transparent and translated requests without rebuilding pools per request. Keep
  buffered, setup and idle limits distinct. Clients cannot raise their bounds;
  auth failures cannot bypass the gateway or trigger credential downgrade; timed-out
  POSTs must not be automatically replayed. A timeout response does not prove
  origin cancellation. Preserve byte-identical default wire behavior and label any
  changed-deadline acceptance cohort separately. This is a design requirement,
  not a shipped configurable feature.

Validation-test audit: neither live harness had direct tests before this follow-up.
The new mixed-harness suite covers rejected/malformed/missing reviews, inclusive
cache accounting, missing artifacts, bad member usage, pytest timeout, startup and
team failures, partial pause results, cancellation and process-state restoration.
Existing coordinator dispatch/approval tests remain the single coverage owner for
those runtime contracts. One redundant constant-string test was removed after the
integration test covered domain injection: before/after coverage retained the same
176 executed script lines and 30 executed branches. No prior runtime tests were
removed.

The opt-in `formation_gateway_matrix.py` ([PR #1149](https://github.com/anvai-labs/victor/pull/1149)) adds artifact scenarios for all twelve
canonical formations and the three ensemble modes, using the existing presets
and coordinator dispatch. Its accounting observer requires physical request,
SQLite, C4 request/session/run, dashboard, and member-usage agreement. Timeouts
remain failures even when a provider subsequently retries successfully. The
[recipe](multiagent-formations-inferflux.md#all-formation-sandhi-matrix) separates
ZAI reference and matched InferFlux runs. Adding the runner does not mark any live
coverage row passed. Its offline tests own scenario configuration, evidence
retention, decision artifacts and accounting fault injection; existing strategy
tests continue to own formation semantics. Removing one duplicate invalid-name
case preserved exactly 78 executed scenario-module lines and 25 branches.

The reflection follow-up combines the redundant context-shim invocation-count
test with its mapping-output test. Before/after coverage preserved exactly 123
reflection-module lines / 36 branches and 733 coordinator lines / 187 branches.
New tests cover opted-in public dispatch, multiple rounds, partial/terminal resume,
failed attempts, malformed verdicts, invalid accounting, snapshot independence and
unchanged behavior for unrelated formations. The earlier failure and duplicate-test
coverage measurements remain separate from live model evidence.


WS-E implementation/evidence: [PR #1115](https://github.com/anvai-labs/victor/pull/1115).

WS-E consumer decisions:

| Contract | Producer | Consumer decision | Compatibility |
|---|---|---|---|
| `team_formation_warning` | StateGraph selector failure | Member sink → client CUSTOM event → v1 wire; UI lanes intentionally ignore, warning log remains visible | Additive; successful/no selector emits none |
| Member `metadata.usage` | SubAgent with `capture_member_usage` | TeamResult and validation harness; Sandhi run tree comparison | Opt-in; absent flag retains old payload |
| TeamResult pause fields | Durable coordinator aggregate | API callers inspect status, paused member, approval request, thread ID | Serialized only when status is present |

Live setup (2026-09-18): LAN access returned. Sandhi at 127.0.0.1:18788 routes
InferFlux through an SSH tunnel on 18080 and ZAI through its coding endpoint. The
[loopback gateway walkthrough](sandhi-zai-loopback.md) covers both providers,
dashboard authentication, usage traces, and reproducible validation commands.

Live follow-up (2026-09-19): the Mac gateway now runs repaired Sandhi `eb38ff4`
through `18081 -> aiserver1:8081`, serving accepted InferFlux `c5d4eb89f` with
partial CUDA offload. The old `18080 -> 8080` route and rollback state remain.
The unchanged Victor harness at `e25fdf751` ran six Qwen members and one ZAI member:
27 HTTP-200 calls, no observed timeouts, 82,837 inclusive input tokens, 18,276
output tokens, and explicit cache reporting for 27/27 requests. ZAI reported
41,664 cache-read tokens; Qwen reported zero. These are reporting observations,
not proof of backend executed reuse. The 120-second buffered deadline remained.
The [recorded verdict](evidence/c5-mixed-gateway-2026-09-19.json) is **FAIL** for
missing artifacts and the review verdict; canonical Victor member-usage
reconciliation was not reached. C5 / [InferFlux #184](https://github.com/anvai-labs/inferflux/issues/184)
remains open pending review and follow-up. This buffered run does not certify
R9700/full Qwen GPU execution, enabled-session leases, origin cancellation,
completed-cache diagnostics or full lifecycle behavior. The earlier five-call
stream pass and original failed evidence were preserved and not repeated.

## 4. Suggested follow-up session plan

Execution record (2026-09-18):

| Workstream | Result | PR |
|---|---|---|
| WS-A | ✅ INTEGRATE orphan trio with all formation surfaces | [#1107](https://github.com/anvai-labs/victor/pull/1107) |
| WS-B | ✅ Consensus, reflection, and partial-failure semantics | [#1108](https://github.com/anvai-labs/victor/pull/1108) |
| WS-C | ✅ Opt-in capacity admission and R9700 validation | [#1110](https://github.com/anvai-labs/victor/pull/1110) |
| WS-D | ✅ Member worktree isolation and wire-session validation | [#1111](https://github.com/anvai-labs/victor/pull/1111) |
| WS-E | ✅ Mixed InferFlux/ZAI review, pause/resume, selection, accounting | [#1115](https://github.com/anvai-labs/victor/pull/1115) |
| WS-F | ✅ Paired battery executed; regressions retained as G32/G33 | [#1118](https://github.com/anvai-labs/victor/pull/1118) |
| WS-G | ✅ FEP-0035, transcript, group chat, debate, handoff | [#1112](https://github.com/anvai-labs/victor/pull/1112), [#1113](https://github.com/anvai-labs/victor/pull/1113) |
| WS-H | ✅ Ensemble vote/judge/synthesizer and R9700 voting | [#1114](https://github.com/anvai-labs/victor/pull/1114) |
| WS-I | ✅ Canonical role terms and supervisor-key normalization | [#1109](https://github.com/anvai-labs/victor/pull/1109) |

WS-F's checkmark means the requested experiment and follow-up filing were completed,
not that either model passed every artifact gate. See its measured matrix above.

Each workstream is independently landable as PRs to develop. WS-A–WS-F are unchanged
from the first cut of this handoff; WS-G/WS-H are new from §2's research; WS-I from
§2.3's nomenclature mandate (it gates WS-G/WS-H API naming — land it first or fold it
into the first WS-G/WS-H PR).

1. **WS-A "Orphan trio decision" (G10)** — cheapest first: decide integrate-vs-archive,
   fix `AdaptiveFormation` stale docstrings either way, wire presets if integrating
   (enum values + `_formations` registration + `AgentTeam` preset + docs update).
   §2.5 strengthens INTEGRATE: router / multi-level-hierarchy / adaptive are exactly
   the three topologies mainstream frameworks ship, and all three are unit-tested
   in-repo.
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
7. **WS-G "Transcript substrate + conversation-native formations" (G14, G15)** —
   FEP first: transcript-backed team context, speaker-selection contract (LLM /
   programmatic / round-robin — mirror AutoGen's `selector_func`/`candidate_func`
   split), termination conditions, peer-handoff message type. Then GROUP_CHAT,
   DEBATE, and SWARM/HANDOFF formations on top. Largest item; do not start without
   the FEP's consumer-decision rows (the consumer-decision rule, §2.4).
8. **WS-H "Ensemble aggregation" (G16)** — substrate-light, can precede WS-G: add an
   `ensemble` aggregation mode to PARALLEL-shaped runs (N samples of one task →
   majority vote / judge / MoA-style synthesizer pass), expose as a CONSENSUS preset
   (`mode="vote"`) and a `create_ensemble_team` preset. Validate on the R9700
   (voting is the pattern most likely to help small local models).
9. **WS-I "Nomenclature standardization" (§2.3 mandate)** — collapse the
   supervisor/manager dual shared-state keys (canonical + deprecated alias), sweep
   docs and formation-role strings to the §2.3 canonical terms, and land the
   critic/judge/reviewer distinction before WS-G/WS-H name their APIs.

## 5. Pointers and sources

- Live-matrix recipe and per-formation examples: [multiagent-formations-inferflux.md](multiagent-formations-inferflux.md)
- Durability contract: [FEP-0028](https://github.com/anvai-labs/victor/blob/develop/feps/fep-0028-team-node-durability-contract.md), ADR-023, TD-25 (roadmap)
- Formation strategies: [`victor/coordination/formations/`](https://github.com/anvai-labs/victor/tree/develop/victor/coordination/formations); coordinator dispatch: `victor/teams/unified_coordinator.py` (`_formations`, `_execute_formation`)
- Session-id derivation: `SubAgentConfig.resolve_member_session_id` (`victor/agent/subagents/base.py`)
- Heterogeneous members: `victor/framework/teams.py` (`TeamMemberSpec`, presets)
- External-harness members: [FEP-0006](https://github.com/anvai-labs/victor/blob/develop/feps/fep-0006-external-harness-executors.md)

### Sources (fetched 2026-09-17)

- LangChain — [Choosing the right multi-agent architecture](https://www.langchain.com/blog/choosing-the-right-multi-agent-architecture) (subagents / skills / handoffs / router; per-pattern communication model and tradeoff tables)
- Anthropic — [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) (prompt chaining, routing, parallelization sectioning/voting, orchestrator-workers, evaluator-optimizer)
- AutoGen — [SelectorGroupChat](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/selector-group-chat.html) and [Swarm](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/swarm.html) (speaker selection, broadcast transcript, handoff-as-tool-call)
- OpenAI Agents SDK — [Handoffs](https://openai.github.io/openai-agents-python/handoffs/) (control transfer, `input_filter`, agent-as-tool contrast)
- CrewAI — [Processes](https://docs.crewai.com/concepts/processes) (sequential, hierarchical with manager_llm/manager_agent)
- Google ADK — [Workflows & multi-agent patterns](https://adk.dev/agents/workflows/) (Sequential/Parallel/Loop workflow agents, LLM transfer, AgentTool; the former multi-agents URL redirects here)
- MDPI Future Internet — [LLM-Based Multi-Agent Orchestration survey](https://www.mdpi.com/1999-5903/18/6/326) (centralized/decentralized/hierarchical + adaptivity dimension; abstract via search)
- arXiv 2601.13671 — [The Orchestration of Multi-Agent Systems](https://arxiv.org/html/2601.13671v1) (orchestration control plane; MCP/A2A substrate roles)
- Anthropic — [How we built our multi-agent research system](https://www.anthropic.com/engineering/built-multi-agent-research-system) (LeadResearcher/subagents/CitationAgent roles; delegation-prompt principles; pass-references-not-payloads; scale heuristics)
- Microsoft Research — [Magentic-One](https://www.microsoft.com/en-us/research/articles/magentic-one-a-generalist-multi-agent-system-for-solving-complex-tasks/) (Orchestrator + specialist roles; TaskLedger/ProgressLedger two-loop mechanics)
- Google — [Announcing A2A](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) (client agent / remote agent terminology; donated to Linux Foundation)
- arXiv 2508.12683 — [A Taxonomy of Hierarchical Multi-Agent Systems](https://arxiv.org/html/2508.12683) (five-axis HMAS taxonomy: control hierarchy, information flow, role/task delegation)
- Classical/paper lineage cited from prior knowledge (stable references): contract-net (Smith 1980), blackboard/Hearsay-II (Nii 1986), multiagent debate (Du et al. 2023), self-consistency (Wang et al. 2022), Mixture-of-Agents (Wang et al. 2024), Magentic-One task/progress ledgers (Microsoft 2024).
