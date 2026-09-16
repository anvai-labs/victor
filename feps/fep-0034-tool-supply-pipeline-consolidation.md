---
fep: "0034"
title: "Tool-Supply Pipeline Consolidation — one per-turn pipeline, one hydration seam, no duplicate selection transports"
type: Standards Track
status: Draft
created: 2026-09-16
modified: 2026-09-16
authors:
  - name: Vijaykumar Singh
    email: vijay@anvaiops.com
    github: vjsingh1984
reviewers: []
discussion: https://github.com/anvai-labs/victor/discussions/0034
---

# FEP-0034: Tool-Supply Pipeline Consolidation

## Summary

The per-turn tool-supply surface (deciding which tools a session sees) has accreted
**three selection entry points with divergent behavior, one dead production path that
held the only demand-hydration hook, and two transports that disagree with each
other**. The dormant-hook failure class has now hit three times (#536 SkillRegistry,
#1057 gh-tool wiring, and the demand-hydration gap this FEP was filed from), each time
producing shipped code that unit tests validated in isolation while no real session
could reach it. This FEP consolidates the surface onto a single, explicitly staged
per-turn pipeline hosted by `ToolSelectionRuntime.select_tools_for_turn`, deletes the
dead path, and unifies the benchmark transport with chat.

## Motivation (verified 2026-09-16)

Live probe on a current develop build: a headless chat session asked to use the `gh`
tool answered *"No dedicated gh tool exists in this session's toolset (git/code/web/
read/write/edit/shell only)"* and fell back to `shell('gh pr list')`, even though
`gh` was demand-wired and registry-level tests passed. Root-cause map:

| Path | Production callers | Demand hydration | Notes |
|------|--------------------|------------------|-------|
| `ToolService.select_tools` | **zero — dead** | ✅ (`_hydrate_tools_for_context`) | the only hydration hook lived here |
| frozen session-tools transport (`get_session_tools()` via `chat_stream_executor`) | **the chat path when cache optimization is on (the default)** | ❌ before the fix | `select_tools_for_turn` is never reached; debug log showed zero tool-supply traces |
| `ToolSelectionRuntime.select_tools_for_turn` | turn executor; chat only when cache optimization is off | ❌ before the fix | curated bypass, Q&A gate, planner, trace |
| `ToolSelector.select_tools` (agent/tool_selection) | benchmark + the runtime internally | ❌ | selector.py:2023 notes the benchmark "never calls select_tools_for_turn — the #353 bug" |

The decisive evidence: a DEBUG-logged live session emitted exactly one supply-related
line — `ToolCatalogLoader: bootstrap-loaded 11 tools` — then
`[cache] Session tools locked: 7 tools (prefix-stable)`, with zero
`ToolSupplyTrace` emissions. The per-turn pipeline this FEP formalizes was
**entirely bypassed** in default chat sessions: the toolset froze at session start,
before any user message could hydrate anything, and the only hydration hook sat on a
path nothing calls.

The cost of the duplication is exactly this: a behavior added at one seam is silent at
the others, and each seam has its own tests, so everything stays green while the
product behavior is broken.

## Design: one staged pipeline (Pipeline pattern)

The per-turn selection entry point is `select_tools_for_turn`; the frozen
transport (`get_session_tools()`) shares the same stage semantics with the
freeze substituting for per-turn re-selection. Their existing implicit stages
become the named, ordered pipeline — each stage is a pure decision over
(turn context, registered-tool set) and emits its outcome to `ToolSupplyTrace`:

1. **DemandHydrate** — register mention-wired tools from the user-message anchor via
   `ToolRegistrar.ensure_tools_for_query` (landed with this FEP's Stage A). Cheap
   no-op when nothing matches or all matches are already registered.
2. **CuratedBypass** — explicit `_enabled_tools` curation returns the stable cached
   list (KV-cache stability, unchanged).
3. **CapabilityGate** — provider/model tool support.
4. **QnAGate** — `tool_supply_policy` three-valued skip decision.
5. **Planner** — `planned_tools` from goals.
6. **Select** — keyword/semantic/hybrid/edge-model selection over the registered set.
7. **Project** — prioritize/filter/schema projection.
8. **TraceFinalize** — `ToolSupplyTrace` emit (already stage-aware).

Rule: **any behavior that changes which tools a session can see must be a pipeline
stage, invoked in every transport before that transport's fork** — never a parallel
hook on a service facade. Demand wiring stays in `SharedToolRegistry` (registry
data), hydration stays a registrar method (registration), but the *invocation* is
one shared helper called at each transport's entry (`hydrate_demand_tools`). Until
Stage C unifies the transports, that is two call sites for one implementation —
deliberately: one call site per transport, one implementation, never one call site
per transport per behavior.

## Staged rollout

- **Stage A (this FEP's companion PR) — DONE** — a single shared
  `hydrate_demand_tools(host, text)` helper (in `tool_selection_runtime.py`)
  called from BOTH transports before their fork: in
  `select_tools_for_turn` (per-turn path) and in `chat_stream_executor`'s
  provider step before the Q&A branch and the session-tool freeze
  (frozen path — a Q&A-classified turn 1 must not lock demand tools out of
  turn 2). Live-verified end-to-end: a headless session asked to use `gh`
  now reports the dedicated gh tool present and calls it. Regression tests:
  `tests/unit/agent/test_demand_hydration_turn.py`. No removals; no behavior
  change for turns that mention nothing demand-wired.
- **Stage B** — delete `ToolService.select_tools` and `_hydrate_tools_for_context`
  (zero production callers; one mock-level unit test in
  `tests/unit/agent/services/test_chat_service.py` moves with it). ToolService keeps
  parse/execute/enabled-tools duties.
- **Stage C** — benchmark transport unification (#353 residue): benchmarks call the
  same pipeline entry (or an explicitly shared stage core) so measured supply
  behavior equals served behavior — the FEP-0025 lesson (measurement must share
  serving's fallbacks) applied to tool supply.
- **Stage D (optional)** — promote stages to first-class objects with per-stage trace
  records if per-stage telemetry demand materializes; not done speculatively.

## Non-goals

- No change to schema caching/KV-stability semantics (CuratedBypass is preserved as-is).
- No change to registry wiring (`BOOTSTRAP_TOOL_SPECS`/`DEMAND_TOOL_SPECS`) — see the
  static-wiring lesson from #1057; that layer is already consolidated.
- No new selection strategies; Stage 6 internals are out of scope.

## Alternatives considered

- **Hydrate in `ToolSelector.select_tools` too** — rejected: two hydration sites
  recreates the divergence this FEP removes.
- **Hydrate at session start only** — rejected: sessions that introduce GitHub (or
  graph) work mid-conversation would never hydrate; per-turn hydration is a string
  scan plus an early-return in the common case.
- **Do nothing; document the three seams** — rejected: the failure class is
  demonstrated, not hypothetical; documentation does not survive the next feature.

## References

- #536 (SkillRegistry dormant hook), #1057 (gh tool wiring), #353 (benchmark
  transport divergence), FEP-0003 (progressive tool loading — adjacent surface),
  FEP-0025 (measurement/serving divergence lesson).
