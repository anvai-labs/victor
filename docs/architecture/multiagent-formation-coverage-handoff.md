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

### 1.5 Completion audit (2026-09-20)

Implementation delivery and live acceptance have different denominators:

| Measure | Complete | Remaining |
|---|---|---|
| WS-A through WS-I increments landed | 9/9 (100%); PRs in §4 | 0 original increments |
| Historical live passes before the new matrix | 7/15 (47%): original six plus ensemble vote | Historical tasks/gates differ; not matched acceptance |
| 2026-09-20 contract-v1 ZAI/Sandhi case passes | 10/15 (67%) | 2 deliverable failures; 3 ensemble cases blocked by resource exhaustion |
| New matched Qwen/InferFlux case passes | 0/15 (0%); not run | 15 cases |
| Combined matched matrix case passes | 10/30 (33%) | 20 cases without a pass (67%); overall ZAI run remains FAIL |
| Current six-Qwen/one-ZAI C5 acceptance | Failed; earlier WS-E success is separate evidence | Full corrected run and reviewed verdict on InferFlux #184 |

The [new actual-member ZAI experiment](evidence/zai-formation-matrix-2026-09-20.json)
passed sequential, parallel, hierarchical, pipeline, consensus, group chat, debate,
handoff, adaptive and dynamic router. Reflection lacked `review.json`; multi-level
hierarchy lacked `second.py`, `test_second.py`, `third.py` and `test_third.py`.
All three ensemble cases failed during file-descriptor exhaustion (G41), so they
do not measure model quality. The original overall verdict remains **FAIL**.
Separate read-only reconciliation matched all 83 calls across 25 sessions against
wire, SQLite, C4 and dashboard, with no further model calls or unrelated ledger
rows. It does not replace the failed verdict or establish lifecycle acceptance.

The union of historical and new passing cases is 12/15 (80%), but spans different
models, tasks and gates: it is coverage evidence, not a matched comparison.
Historical WS-F strict acceptance remains Qwen 1/6 and LFM 0/6. No weighted overall
completion percentage is asserted, and case percentages do not estimate effort.
The remaining correctness work includes G32 completion, G34 buffered reporting,
G39 task binding, G41 resource exhaustion and G43 member-pytest cleanup.
G36 structured verification landed in [PR #1159](https://github.com/anvai-labs/victor/pull/1159)
after all CI gates passed, including Vertical Py3.12. G42 independent numeric oracles landed in
[PR #1157](https://github.com/anvai-labs/victor/pull/1157) with all CI green, including
Vertical Py3.12. No contract-v2 cases have run; do not combine v1/v2 pass counts. The [research evaluation](multiagent-formation-research-evaluation.md)
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

- **G18 — ROCm InferFlux does not publish sequence capacity.** Verified live during
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

- **G39 — rewritten formation tasks can omit declared member assignments.**
  The coordinator's SubAgent adapter suppresses the static member goal whenever
  the dispatched task differs from the team goal, treating that difference as
  dynamic delegation. Conversation and other structured formation tasks also
  differ, so a member can receive its response contract without its declared
  artifact assignment. An offline GROUP_CHAT reproduction returned success while
  the captured SubAgent task omitted the unique member assignment. This is an
  offline task-binding finding, not evidence of model quality or a live pass.
  The opt-in gateway matrix explicitly carries assignments keyed by the preset's
  canonical member IDs inside its structured team objective. Existing coordinator
  defaults are unchanged; a general task-binding correction remains open.
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

- **G43 — member-pytest cleanup can exceed its execution deadline.** The matrix's
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
