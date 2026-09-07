---
fep: "0031"
title: "Chat Runtime Inversion — ChatService owns the turn lifecycle"
type: Standards Track
status: Draft
created: 2026-09-06
modified: 2026-09-06
authors:
  - name: Vijaykumar Singh
    email: vijay@anvaiops.com
    github: vjsingh1984
reviewers: []
discussion: https://github.com/anvai-labs/victor/discussions/0031
---

# FEP-0031: Chat Runtime Inversion — ChatService owns the turn lifecycle

## Summary

Victor's chat turn lifecycle is split across three layers with the **orchestrator owning the
glue**: `ChatService` frames the turn and immediately calls back into eight orchestrator-supplied
handlers (`bind_runtime_components`); `TurnExecutor` (service layer) already builds and drives the
`AgenticLoop`; and `AgentOrchestrator` — the documented *facade* — still supplies every runtime,
owns turn setup/teardown, task-report framing, tool selection, and context-limit handling, and is
reached around **53 sites calling `orch._*` privates** (55 occurrences) in the chat streaming
cluster (4,349 lines across `chat_stream_runtime.py` / `chat_stream_executor.py` /
`chat_stream_helpers.py` / `streaming_act_adapter.py`), touching **29 distinct facade
internals**, with `TurnExecutor` reaching back via `_resolve_orchestrator()` at 10 call sites. This FEP inverts that:
ChatService owns the turn lifecycle end-to-end, receives frozen per-turn state instead of
setup/teardown handler pairs, and the chat runtime cluster depends on a narrow, explicitly
enumerated `ChatRuntimeServices` view instead of the orchestrator's privates. Public contracts
(`Agent.run`/`Agent.stream`, `orchestrator.chat()`'s facade methods) are unchanged.

## Motivation

### The finding (U1-4 / U1-7, co-design review 2026-09-03)

> **U1-4**: "ChatService is nominal: `bind_runtime_components` injects 8 orchestrator handlers;
> 'runtimes' are built with the orchestrator itself and call facade privates
> (`orch._parse_and_validate_tool_calls`)" → *"Invert: ChatService owns turn lifecycle, receives
> frozen per-turn state; move `_prepare/_teardown_chat_service_turn_runtime` into ChatService."*
>
> **U1-7**: "God-file persists: 4,225 LOC, ~230 defs, 9 `_initialize_*` methods, 115
> `getattr(self,…)`, 322 commits/12m, fan_in=39" → *"Ratchet guard on def-count; move property
> blocks (~800 lines) to mixins."*

The review's sequencing requirement — upgrade the guards **first** — is satisfied: #1023 added the
AST-based facade guard plus a ≤8 kwonly ratchet on `bind_runtime_components`, and #1024 added the
orchestrator structural ratchet (line cap 4,225; def-count ≤226; self-probe counts ≤136 getattr /
≤24 hasattr).

### The measured problem

- **Churn×size leader.** `orchestrator.py` remains the repo's top churn×size file
  (4,225 lines, 322 commits/12mo, fan_in 39) — every chat-path change pays the god-file tax.
- **The inversion seam exists but inverts nothing.** All eight `bind_runtime_components` handlers
  (bound at `orchestrator.py:762-774`) resolve to orchestrator-owned code: five are facade
  methods/properties directly (`turn_executor` lazy proxy, `task_report_start/finish`,
  `turn_setup/teardown`), and three are methods of helpers the orchestrator constructs from
  itself (`PlanningChatRuntime(self)`, `ContextLimitRuntime(self)`, the stream-adapter factory).
  ChatService calls back into the facade's orbit for every substantial step.
- **Privates leak across the service boundary.** 53 `orch._*` call sites (55 occurrences) in
  the chat streaming cluster reach into 29 distinct facade internals — most-touched: `_chunk_generator` (×7),
  `_required_outputs`/`_required_files` (×5 each), `_tool_planner`, `_context_manager`,
  `_task_completion_detector`, `_select_tools_for_turn`, `_conversation_controller`,
  `_context_compactor`. `TurnExecutor._resolve_orchestrator()` reaches back at 10 call sites.
- **Cost center is the cluster, not ChatService.** `chat_service.py` itself (1,678 lines) is
  clean — framing + metrics, no facade reach-through. The tangle lives in the four
  `chat_stream_*`/adapter files (4,349 lines combined), alongside `TurnExecutor`
  (`turn_execution_runtime.py`, 2,376 lines, already service-layer).

### What is *already* inverted (verified 2026-09-06)

Less than the finding implies, and more than a rewrite would need: `TurnExecutor`
(`turn_execution_runtime.py`, 2,376 lines, service layer) already **builds and drives the
`AgenticLoop` itself** (`_execute_via_agentic_loop`, :1051; loop construction :1104) — the loop
does not live in the orchestrator. Post-FEP-0007, streaming loop ownership is likewise the
framework `AgenticLoop` via `StreamingActAdapter`. What remains inverted-in-name-only is
**ownership of the lifecycle frame** (who sets up, supplies collaborators, and tears down a turn)
and the reach-through web. This FEP is therefore a *sever-and-rehome* refactor, not a rewrite.

## Proposed Change

### 1. ChatService owns the turn frame

`_prepare_chat_service_turn_runtime` / `_teardown_chat_service_turn_runtime` and
`_start_task_report` / `_finish_task_report` move from the orchestrator into ChatService (or a
`ChatTurnRuntime` component it constructs); `_handle_context_and_iteration_limits` is already a
pure passthrough to `ChatService.handle_context_and_iteration_limits` — its facade shim is
deleted rather than moved. The
`turn_setup_handler` + `turn_teardown_handler` pair collapses into ChatService-internal calls;
the ≤8 kwarg ratchet on `bind_runtime_components` is allowed to **shrink** to 6.

### 2. `ChatRuntimeServices`: a narrow, enumerated collaborator view

The 29 reached-through facade internals are replaced by one explicitly enumerated protocol/view —
`ChatRuntimeServices` — constructed once at bind time and passed to the chat runtime cluster.
Each entry is declared with its minimal surface (e.g. `tool_planner.plan()`, not the planner
object; `chunk_generator.for_provider(...)`, not `orch._chunk_generator`). Adding an entry
requires editing the protocol — and the AST facade guard (#1023) gains a check that the chat
runtime cluster never holds or touches the orchestrator object at all (walking `ast.Attribute`
chains rooted at any orchestrator alias, since today's reach-throughs ride in on
`orch = self._orchestrator` bindings, not imports), so the reach-through count can only fall,
never regrow.

### 3. Rehome the handler-supplier factories

`_get_planning_chat_runtime`, `_get_context_limit_runtime`, `_get_chat_stream_adapter`, and
`_select_tools_for_turn` move out of the orchestrator into the chat runtime cluster. The
orchestrator keeps the public facade methods (`chat`, `chat_with_planning`, `stream_chat` →
ChatService): `chat` and `stream_chat` already carry explicit deprecation markers;
`chat_with_planning` (:3189) is an undeclared pure passthrough and gains one before any
retirement — the shims collectively retain ~13 production call sites, so retirement includes a
caller-migration sub-step.

### 4. Property blocks → mixins (U1-7)

~800 lines of orchestrator property blocks move to mixin modules under
`victor/agent/runtime/`, shrinking the facade file without changing the public class.

### 5. Consolidate the chat runtime cluster

The four `chat_stream_*`/adapter files (4,349 lines) move under
`victor/agent/services/chat_runtime/` as a package owned by ChatService — file moves +
`ChatRuntimeServices` rewiring only, no logic changes (FEP-0007's characterization discipline:
the streaming battery and parity battery stay byte-green).

## Implementation Plan

Each phase is independently landable and guard-gated.

1. **Introduce `ChatRuntimeServices`** with the enumerated view; mechanically rewire the 53
   `orch._*` sites onto it (pure rename-through-a-protocol, one PR per cluster file).
   Gate: facade-guard extension proves zero remaining `orch._*` in the cluster; all batteries
   byte-green.
2. **Move the turn frame into ChatService** (setup/teardown + task-report + context-limits);
   collapse the setup/teardown handler pair; shrink `bind_runtime_components` to 6 kwargs.
   Gate: service-validation suite + streaming battery green; kwarg ratchet lowered 8 → 6.
3. **Rehome the handler-supplier factories** and retire the orchestrator chat shims:
   `chat` and `stream_chat` carry explicit deprecation markers; `chat_with_planning` is an
   undeclared pure passthrough today and gains one before any removal. The shims retain ~13
   production call sites (integrations routes, subagents, planning, evaluation adapters), so
   retirement includes a caller-migration sub-step.
   Gate: structural ratchet **lowered** (line cap 4,225 → ~2,500; def_count 226 → ~180; probe
   counts down), never raised.
4. **Property mixins + cluster package consolidation** (U1-7). Gate: batteries green; facade
   file no longer the churn×size leader on the next review pass.

## Benefits

- The facade is finally *facade-only* in fact, not just docstring — the orchestrator stops
  owning chat behavior, cutting the god-file tax on every chat change.
- The service boundary becomes checkable: enumerated `ChatRuntimeServices` + AST guard means
  boundary erosion fails CI instead of accumulating silently.
- Single owner for the turn lifecycle makes FEP-0029 (durable chat continuation) and future
  resume/interrupt work (item 24) tractable — both need exactly this ownership.
- The ratchets turn from ceiling-pressure into demonstrated progress (caps only ever lowered).

## Drawbacks and Alternatives

- **Large mechanical surface.** 53 rewiring sites + file moves; mitigated by per-file PRs and
  byte-green battery gates (the FEP-0007 discipline).
- **`ChatRuntimeServices` could become a god-view.** Mitigated by declaring minimal per-entry
  surfaces and forbidding pass-through of the orchestrator itself.
- **Alternative (rejected): leave as-is, rely on ratchets.** The ratchets cap growth but don't
  reduce the reach-through web; U1-4's failure mode (services calling facade privates) persists.
- **Alternative (rejected): full turn-lifecycle rewrite** (event-sourced turn state machine).
  Vastly more risk for the same ownership goal; the existing `TurnExecutor`/`AgenticLoop`
  machinery is sound — only its wiring is inverted.

## Unresolved Questions

- Does `ChatTurnRuntime` live as part of ChatService or as a sibling service component
  constructed by it? (Lean: sibling component, ChatService constructs and owns it — avoids
  growing `chat_service.py` past its own readability.)
- The orchestrator chat shims have ~13 remaining production call sites; phase 3 includes the
  caller migration, but the exact split (migrate callers to `Agent`/`ChatService` vs keep thin
  facade forwarders permanently) is settled during phase 3 review.

## Migration Path

Behavior-neutral end to end: public `Agent.run`/`Agent.stream`, `orchestrator.chat()`,
`ChatService.chat/stream_chat` signatures unchanged; `bind_runtime_components` shrinks but its
callers are all inside the orchestrator's own bind site. Each phase lands behind green batteries
(streaming characterization, run/stream parity, service validation, structural ratchets) with
ratchet caps **lowered** at the phase that earns it.

## Compatibility

No public API change. The `TurnExecutor` protocol and `AgenticLoop` injection surface are
untouched (the loop already takes everything it needs). Downstream verticals and SDK consumers
see no difference; the change is internal to `victor/agent`.

## Acceptance Criteria

- Zero `orch._*` references in the chat runtime cluster (AST-guard-enforced, not convention).
- `bind_runtime_components` kwonly params ≤ 6 (ratchet lowered from 8).
- Orchestrator ratchets lowered and pinned: ≤ ~2,500 lines, def_count ≤ ~180, probe counts
  materially below 136/24.
- Streaming characterization battery and run/stream parity battery byte-green at every phase.
- `docs/reviews/2026-09-03-codesign/README.md` items 27 marked done with the PR map.

## References

- U1-4/U1-7 — `docs/reviews/2026-09-03-codesign/U1-agentic-core.md` (findings + inversion sketch).
- #1023 — AST facade guard + `bind_runtime_components` ratchet (prerequisite, landed).
- #1024 — orchestrator structural ratchet (prerequisite, landed).
- FEP-0007 (Implemented) — unified loop + `StreamingActAdapter`, the extraction-seam precedent.
- FEP-0029 — durable chat continuation; consumer of this FEP's turn-lifecycle ownership.
