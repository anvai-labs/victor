# Multi-Agent Formation Coverage, Gaps, and Handoff

**Date:** 2026-09-17 · **Status:** Handoff for a follow-up session · **Live matrix:** R9700 / InferFlux / Qwen3-Coder-30B (see [multiagent-formations-inferflux.md](multiagent-formations-inferflux.md)) · **Rev 2:** adds §2, an industry pattern catalog researched from current framework docs and surveys (LangChain, Anthropic, AutoGen, OpenAI, CrewAI, ADK, MDPI/arXiv), with three new gaps (G14–G16) and two new workstreams (WS-G/WS-H). · **Rev 3:** adds §2.3 role-nomenclature standardization, §2.4 cross-cutting standards mandate (prompt engineering, design patterns, architecture), and workstream WS-I; adversarially reviewed — all codebase claims re-verified against the repo.

This document answers three questions: which formations Victor implements and which of
those were verified live; which designs exist beyond the canonical six (implemented,
unwired, or deferred); and what the follow-up session should address, with the
co-design learnings that motivate each item.

## 1. Coverage: what is implemented, and what was verified live

### 1.1 Canonical formations — implemented and live-tested (6/6)

`TeamFormation` ([`victor/teams/types.py`](../../victor/teams/types.py)) defines twelve values, all registered in `UnifiedTeamCoordinator._formations`.
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
  (heterogeneous-member and review-preset suites). WS-E adds a ZAI-only gateway
  live harness; cross-vendor local/cloud validation remains pending LAN recovery.
- **Member-granular durability** (FEP-0028 / ADR-023 / TD-25, shipped #733–#752):
  per-member checkpoint/resume at formation-natural granularity across all six,
  durable `MemberApprovalPause` for the four non-iterative formations, and
  `MemberEventSink` per-member streaming lanes. WS-E validates an injected approval
  pause/resume on ZAI; the requested local-inference rerun remains pending.

### 1.3 Additional formations — WS-A integration

**Decision: INTEGRATE.** The orphan trio now has enum values, shared-registry
registration, `AgentTeam` presets, feature/formation docs, coordinator-dispatch tests,
and explicit `supports_durable_pause() == False` statements. Live validation remains
outstanding; none of these rows claims a live pass.

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
- **G7 — cross-vendor live validation pending.** The ZAI-only Sandhi gateway and
  review preset are exercised by the WS-E harness. Local InferFlux workers plus
  a cloud reviewer, including InferFlux reasoning-effort stripping, await LAN
  recovery; the user explicitly restricted current runs to ZAI on 2026-09-18.
- **G8 — local-inference durability validation pending.** WS-E covers an injected
  mid-PIPELINE `MemberApprovalPause`, checkpoint, and resume over ZAI. The completed
  writer must not rerun. InferFlux replay remains pending; FEP-0028 non-goals remain.
- **G9 — ✅ observable dynamic selection (WS-E).** Exceptions and invalid formation
  identifiers emit warning logs plus `team_formation_warning` through the member
  sink, client stream, and v1 wire. The ZAI harness asserts PARALLEL selection and
  warned default dispatch. Async selectors retain explicit TypeError rejection.
- **G10 — ✅ orphan trio: INTEGRATE (WS-A, [PR #1107](https://github.com/anvai-labs/victor/pull/1107)).** All three have public enum,
  registry, preset, docs, dispatch-test, and durability surfaces. Adaptive stale
  names and duplicate dispatch were removed. Regression tests cover real member
  execution/failure, concurrent adaptive calls, lossless task splitting, and invalid
  trees. Original six defaults remain unchanged.
- **G11 — ✅ ZAI member accounting validation; R9700 pending (WS-E).** Opt-in `capture_member_usage`
  exposes neutral counters in member metadata; retry totals preserve all attempts.
  Seven members passed exact input/output/total reconciliation against each Sandhi
  run tree; fourteen files and seven independent pytest tests passed in 167.93s.
  The R9700-specific reconciliation remains pending LAN recovery.
- **G12 — ✅ formation-aware tool supply documented (WS-C, [PR #1110](https://github.com/anvai-labs/victor/pull/1110)).** Member `allowed_tools`
  narrows the registry before provider supply. The live capacity run supplied four
  filesystem/shell tools per member; global pruning defaults remain unchanged.
- **G13 — guard tuning is a two-model sample.** Narration/intent/refusal classifiers
  were tuned on Qwen3-Coder-30B + GLM-5.3 only. Before claiming edge-model support,
  run the same matrix on a small local model (qwen3.5:2b class) and record deltas.
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

- **G22 — ✅ public pipeline pause metadata loss (WS-E).** The spawn adapter dropped
  `awaiting_approval` and `approval_request`, and TeamResult dropped aggregate pause
  fields. Both boundaries now preserve the structured signal. Absent pause state,
  TeamResult serialization remains unchanged.
- **G23 — ✅ usage writes mutated snapshots (WS-E).** SessionStateAccessor returned
  a copy while runtime writers and metrics expected a live accumulator. Internal
  access now preserves one dictionary's identity, including assignment; the public
  SessionStateManager snapshot remains defensive. Real-runtime regression tests
  cover inclusive input, output, cache, reasoning, and reset visibility.
- **G24 — ✅ explicit member overrides bypassed gateways (WS-E).** Overrides now use
  the existing canonical gateway resolver for configured provider blocks/environment,
  pass the gateway to the managed factory, and use its virtual key. Invalid explicit
  gateway setup fails closed. Legacy direct-provider warned inheritance remains.
- **G25 — Sandhi transparent ZAI requests omitted JSON Content-Type.** Fixed and
  live-tested in [Sandhi PR #265](https://github.com/anvai-labs/sandhi/pull/265);
  23 raw-forwarding tests passed and all CI gates are green. Repository policy
  requires an approving review before merge; the patched local binary is in use.

WS-E consumer decisions:

| Contract | Producer | Consumer decision | Compatibility |
|---|---|---|---|
| `team_formation_warning` | StateGraph selector failure | Member sink → client CUSTOM event → v1 wire; UI lanes intentionally ignore, warning log remains visible | Additive; successful/no selector emits none |
| Member `metadata.usage` | SubAgent with `capture_member_usage` | TeamResult and validation harness; Sandhi run tree comparison | Opt-in; absent flag retains old payload |
| TeamResult pause fields | Durable coordinator aggregate | API callers inspect status, paused member, approval request, thread ID | Serialized only when status is present |

Current live constraint (2026-09-18): router down; user requested **ZAI-only through
Sandhi**. Do not mark G7/G8's R9700 steps or WS-F/G13 complete from cloud-model data.
The [loopback gateway walkthrough](sandhi-zai-loopback.md) covers the active setup.

## 4. Suggested follow-up session plan

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
- Durability contract: [FEP-0028](../../feps/fep-0028-team-node-durability-contract.md), ADR-023, TD-25 (roadmap)
- Formation strategies: [`victor/coordination/formations/`](../../victor/coordination/formations/); coordinator dispatch: `victor/teams/unified_coordinator.py` (`_formations`, `_execute_formation`)
- Session-id derivation: `SubAgentConfig.resolve_member_session_id` (`victor/agent/subagents/base.py`)
- Heterogeneous members: `victor/framework/teams.py` (`TeamMemberSpec`, presets)
- External-harness members: [FEP-0006](../../feps/fep-0006-external-harness-executors.md)

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
