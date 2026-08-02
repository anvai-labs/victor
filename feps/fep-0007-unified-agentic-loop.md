---
fep: 0007
title: "Unified Agentic Loop (single loop, two I/O modes)"
type: Standards Track
status: Draft
created: 2026-06-21
modified: 2026-06-21
authors:
  - name: Vijaykumar Singh
    email: vijay@anvaiops.com
    github: vjsingh1984
reviewers: []
discussion: https://github.com/anvai-labs/victor/discussions/0007
---

# FEP-0007: Unified Agentic Loop

## Summary

Victor runs **two** agentic iteration loops:

- `AgenticLoop.run()` (`victor/framework/agentic_loop.py`) — the headless/buffered loop for
  `victor chat`. A formal phased loop: PERCEIVE → PLAN → ACT → EVALUATE → DECIDE (named
  `LoopStage`s), returning a `LoopResult`.
- `StreamingChatExecutor.run()` (`victor/agent/services/chat_stream_executor.py`) — the live
  streaming loop for `victor ui` / `Agent.stream()`, created by
  `ChatStreamRuntime.get_executor()` (`chat_stream_runtime.py:159`). A procedural loop that
  yields `StreamChunk`s token-by-token. It has **no PLAN/DECIDE phase**, receives perception
  injected from upstream rather than running a PERCEIVE phase, and reimplements the per-turn
  EVALUATE guards inline.

The **only fundamental difference is I/O timing** — buffered `LoopResult` vs streamed
`AsyncIterator[StreamChunk]`. Everything else (perceive, the per-turn continue/nudge/stop
decision) is conceptually the same and was, until recently, duplicated. This FEP proposes
collapsing the two into **one loop with two I/O modes**, so PERCEIVE / PLAN / EVALUATE /
DECIDE exist once and only ACT differs (buffered turn vs streamed turn).

## Motivation

- **Maintenance surface.** Every loop-quality fix (loop guards, convergence, recovery) has had
  to be written twice and has **drifted** — e.g. content-repetition detection used a cruder
  content-length heuristic in the headless loop and a better hash/overlap detector in the
  streaming loop. The plateau formula and nudge activation also diverged.
- **Capability gap.** The streaming loop lacks PLAN/DECIDE and the structured fulfillment /
  convergence machinery the headless loop has, so the UI path is "thinner" and behaves
  differently from the CLI path for the same task.
- **Target architecture.** CLAUDE.md states the runtime is service-first and "AgenticLoop is
  the canonical execution authority for chat," and explicitly notes "the streaming path
  (StreamingChatPipeline) has its own iteration loop and is not yet integrated with
  AgenticLoop." This FEP closes that gap.

## Prerequisite (landed)

PR #170 extracted the per-turn EVALUATE guards (content-repetition, plateau, spin, nudge) into
a shared `TurnEvaluationController` in `victor/agent/turn_policy.py` that **both** loops call.
This FEP builds on that controller as the shared decision seam.

## Implementation Status & Decision (2026-06-21)

Phases 1–2 have landed, and a key delivery decision was made that **supersedes the flag-gated
Phase 3/4 rollout originally drafted below**:

> **Decision: no feature flag, no legacy fallback.** The unified per-turn primitive is the
> *single canonical* streaming loop body, on by default. A `USE_UNIFIED_STREAMING_LOOP` flag
> with the old loop retained as a fallback would be exactly the branch-proliferation / dual-path
> tech debt the runtime guidance pushes against (cf. the W3 service-layer flag removal). We
> guard the refactor with a byte-stable characterization battery instead of a runtime A/B.

Shipped:

- **Phase 1 (EVALUATE consolidation)** — the streaming loop's per-turn decision (content-repetition,
  plateau, spin, nudge, search-novelty, fulfillment) routes through the shared
  `TurnEvaluationController` / `turn_policy` seam.
- **Phase 2 (`stream_turn()` extraction + assembly)** — the per-turn body of
  `StreamingChatExecutor.run()` was decomposed into named ACT/EVALUATE/emit/recovery/continuation
  helpers (PRs #176–#190) and then **assembled into a single `_stream_turn()` async-generator
  primitive** (PR #192); `run()` is now a thin while-driver over it. A deterministic, offline
  streaming **characterization battery** (`tests/integration/streaming/`, PRs #186/#188) pins the
  loop's observable behavior on the QA scenarios (S1/S2/M1/M2/C1/W1–W4/U1) and gated every step
  byte-stable. The flag-gated comparison scaffolding was removed once the no-flag decision was made
  (PR #193).

Remaining (now in progress): the full **capability-leveling** unification — make `AgenticLoop` the
single PPAED loop for both modes via a symmetric streaming-ACT port, so the UI path gains
PERCEIVE/PLAN/DECIDE and the research-rooted gates it currently lacks. The owner decided on
2026-06-21 to pursue this (framing **B**) rather than close the FEP; the concrete architecture,
phased plan, and the revised run/stream behavioral-parity acceptance bar are in **Addendum A**.

## Proposed Change

Make `AgenticLoop` own the iteration for both modes:

```python
class AgenticLoop:
    async def run(self, query, ...) -> LoopResult:            # buffered (today)
    async def run_streaming(self, query, ...) -> AsyncIterator[StreamChunk]:  # NEW
```

Both drive the same phase sequence and the same `TurnEvaluationController`. They differ only in
the ACT primitive:

- buffered ACT → `_act()` → `ExecutionCoordinator.execute_turn_with_tools()` (returns a
  `TurnResult`).
- streaming ACT → a new `stream_turn()` primitive extracted from `StreamingChatExecutor.run()`
  that yields `StreamChunk`s **and** produces the same per-turn observation
  (`TurnObservation`) the EVALUATE phase needs.

`StreamingChatExecutor` is reduced to: (a) the `stream_turn()` per-turn primitive (provider SSE
+ tool execution + recovery for ONE turn) and (b) a thin governance/adapter shell.
`ChatStreamRuntime.get_executor()` is repointed to drive `AgenticLoop.run_streaming()`.

## Implementation Plan

Each phase is independently landable and revertible.

- **Phase 1 — finish EVALUATE consolidation (no structural risk).** Route the streaming loop's
  remaining inline guards (plateau, and later fulfillment) through `TurnEvaluationController`
  so the *entire* per-turn decision is shared, not just content-repetition. Unify the
  perception injection so both loops perceive via the same path. *No FEP gate — extends PR
  #170.*
- **Phase 2 — extract and assemble `stream_turn()`. ✅ Done.** Pull the per-turn streaming (one
  provider stream + tool execution + recovery, yielding chunks and signalling the per-turn outcome)
  out of `StreamingChatExecutor.run()`, first as named helpers and then as a single `_stream_turn()`
  primitive. Behavior-neutral: `run()` is a thin while-driver that calls it. Locked byte-stable by
  the streaming characterization battery.
- **Phase 3 — drive the single per-turn primitive (no flag, no fallback).** The unified
  `_stream_turn()` is the one canonical streaming loop body, on by default — there is no
  `USE_UNIFIED_STREAMING_LOOP` gate and no retained legacy loop. (Optional deeper unification: have
  `AgenticLoop` drive `_stream_turn()` as its streaming ACT so PERCEIVE/EVALUATE/DECIDE live once
  across both modes — still as the single implementation, no flag, no fallback.)
- **Phase 4 — n/a (folded into the no-flag approach).** There is no separate "flip the flag /
  delete the old loop" step: the old iteration loop was replaced in place by the `_stream_turn()`
  driver rather than kept behind a flag, so there is no dual path to cut over from.

## Benefits

- One place for loop control, perceive, evaluate, decide, convergence, and recovery.
- The streaming UI path gains the headless loop's PLAN/DECIDE + fulfillment/convergence.
- No more drift between CLI and UI behavior for the same task.

## Drawbacks and Alternatives

- **Risk.** The refactor touches the live UI streaming path; a regression breaks `victor ui`.
  Mitigated **not** by a feature flag + fallback, but by a deterministic, offline characterization
  battery that drives the real loop and pins every QA scenario byte-stable, plus live-provider
  verification, on each incremental change.
- **Alternative (rejected): leave two loops, share only guards.** PR #170 already did the
  cheap part; leaving the structural split keeps two iteration controls, the missing
  PLAN/DECIDE in streaming, and ongoing drift risk.
- **Alternative (rejected): make `StreamingChatExecutor` canonical.** It lacks the phase
  structure and is service-layer, not framework; CLAUDE.md designates `AgenticLoop` canonical.

## Unresolved Questions

- **Chunk granularity. ✅ Resolved in Phase 2.** Token/segment-granularity emission stayed inside
  the provider/tool helpers (`_stream_provider_turn`, `_execute_tools_turn`); `_stream_turn()` is one
  loop iteration that re-yields those chunks. Loop control crosses the generator boundary via a
  small mutable `_TurnOutcome` holder (`should_stop` + the inter-turn separator state) rather than a
  return value.
- **Recovery placement. ✅ Resolved in Phase 2.** Per-turn recovery (`_apply_turn_recovery`) and the
  empty-response ladder (`_emit_assistant_turn`) live inside the per-turn primitive; loop-level
  control (continue/stop) is signalled out via the helper holders. The cross-task cleanup hardened
  in PR #155 stays loop-level and was kept byte-stable by the characterization battery.
- **StateGraph mode interaction.** `USE_STATEGRAPH_AGENTIC_LOOP` swaps the inner *headless* executor
  and is independent of the streaming primitive. Only relevant if optional deeper unification routes
  the headless loop through `_stream_turn()`; can be deferred.

## Migration Path

The change is behavior-neutral and in-place (no flag, no parallel path), so there is no breaking
step for callers:

1. Phases 1–2 landed with no behavior change (EVALUATE consolidation + the `_stream_turn()`
   extraction and assembly); the streaming characterization battery is the regression gate.
2. The unified `_stream_turn()` primitive replaced `StreamingChatExecutor.run()`'s iteration body
   **in place** — `run()` became a thin driver over it. There is no dual path and nothing to opt
   into: the single canonical streaming loop is always active.
3. Each incremental change was held byte-stable by the QA battery (S1/S2/M1/M2/C1/W1–W4/U1) plus
   live-provider verification (same answers, no streaming tracebacks).
4. `StreamChunk` (the public streaming type) and the `Agent.stream()` async-iterator contract are
   unchanged throughout, so UI, API, and SDK consumers need no migration.

## Compatibility

`AgenticLoop.run()` and the buffered path are unchanged. The streaming refactor is internal to
`StreamingChatExecutor` and behavior-neutral, so existing callers keep the current streaming
behavior. `StreamChunk` (the public streaming type) and the `Agent.stream()` async-iterator contract
are unchanged throughout, so UI, API, and SDK consumers require no code changes.

## Acceptance Criteria

- The unified `_stream_turn()` primitive yields `StreamChunk`s byte-identical to the pre-refactor
  `StreamingChatExecutor.run()` on the QA battery (S1/S2/M1/M2/C1/W1–W4/U1) — same answers, same
  tool sequence, no streaming tracebacks. *(Met: the characterization battery passes byte-stable.)*
- The streaming per-turn decision shares one `TurnEvaluationController` / `turn_policy` seam.
- Single canonical streaming loop — no feature flag, no retained legacy path.

## Addendum A — Decision: full capability-leveling unification (2026-06-21)

Phases 1–2 met FEP-0007's *minimum* goal (one canonical streaming per-turn primitive + a shared
per-turn decision). This addendum records the decision to pursue the **maximal** goal instead of
closing the FEP: make the research-rooted `AgenticLoop` (PERCEIVE → PLAN → ACT → EVALUATE → DECIDE)
the **single loop for both modes**, with streaming reduced to a pure I/O concern.

> **Decision (owner, 2026-06-21): pursue framing (B), capability-leveling.** The phase loop is the
> scientifically grounded execution model (it carries five arXiv-backed mechanisms; see below).
> Streaming vs. buffered is a UI/UX delivery difference, **not** a behavioral one — the live UI path
> should run the *same* PPAED loop and merely emit `StreamChunk`s as it goes. The streaming path's
> current absence of PLAN / DECIDE / planning-gate / semantic-cache / routing is a capability gap to
> *close*, not a behavior to preserve. Consequently the acceptance bar is **run/stream behavioral
> parity**, not byte-stability of today's streaming output (see "Acceptance bar" below).

This supersedes the earlier "leaning no-go" recommendation, which optimized for behavior-neutrality;
the owner has explicitly chosen feature-completeness of the UI path over preserving its current
(thinner) behavior.

### Exploration findings (2026-06-21)

A three-way read of `agentic_loop.py`, `chat_stream_executor.py`, and the I/O/caller surface
established:

| Concern | Headless `AgenticLoop.run()` | Streaming `StreamingChatExecutor.run()` |
|---|---|---|
| I/O shape | buffered `LoopResult` → `CompletionResponse` | `AsyncIterator[StreamChunk]` |
| Phase methods | `_analyze_turn` / `_plan` / `_act` / `_evaluate` / DECIDE — **non-yielding**; only ACT does I/O | one procedural `_stream_turn()` async-gen |
| ACT seam | **injected `turn_executor.execute_turn()` → `TurnResult`** (agentic_loop.py:2069) | inline `_stream_provider_turn` + `_emit_assistant_turn` + `_execute_tools_turn` |
| EVALUATE/DECIDE | full structured phases (enhanced-completion, fulfillment, adaptive termination) | thin, re-implemented inline; emits a final chunk on terminal |
| PERCEIVE | per-turn (`runtime_intelligence.analyze_turn`) | one-time upfront injection only |
| PLAN / DECIDE / fast-slow gate / semantic cache / paradigm routing | present | **absent** |
| Shared decision seam | `TurnEvaluationController.evaluate(TurnObservation)` | `TurnEvaluationController.evaluate(TurnObservation)` ✅ already shared |

**The enabling insight:** `AgenticLoop` already takes `turn_executor` as an injected dependency and
calls `turn_executor.execute_turn()` as its buffered ACT. Streaming ACT therefore slots into the
**same injection seam** — symmetric with the buffered path — so the framework loop never has to
absorb service-layer streaming plumbing (`orch._chunk_generator`, `stream_ctx`, governance gates).
This keeps the CLAUDE.md layering intact: framework owns the phase loop + decision; the service
owns the ACT *effect*.

### Chosen architecture

`AgenticLoop` owns iteration for both modes; the **only** per-turn difference is the ACT call:

```python
class AgenticLoop:
    async def run(self, query, ...) -> LoopResult:                          # buffered (today)
        ...
        result = await self.turn_executor.execute_turn(...)                 # buffered ACT
        evaluation = await self._evaluate(result, ...)                      # shared, non-yielding
        decision   = self._decide(evaluation, ...)                          # shared
        ...

    async def run_streaming(self, query, ...) -> AsyncIterator[StreamChunk]:  # NEW
        # SAME _analyze_turn / _plan / _evaluate / _decide calls — all non-yielding
        result = None
        async for item in self.turn_executor.execute_turn_streaming(...):    # streaming ACT
            if isinstance(item, StreamChunk):
                yield item
            else:
                result = item                                                # final TurnResult
        evaluation = await self._evaluate(result, ...)                       # shared, non-yielding
        decision   = self._decide(evaluation, ...)                          # shared
        if decision.terminal:
            yield self._final_chunk(decision); return
        yield self._nudge_chunk(decision)                                    # DECIDE → emits a chunk
```

- **New ACT port:** `turn_executor.execute_turn_streaming(...)` — an async generator that yields
  `StreamChunk`s for live token + tool output **and** produces the same `TurnResult` the EVALUATE
  phase consumes (delivered as the terminal item, or via an out-param holder, mirroring the existing
  `_stream_turn` holder pattern). It is assembled from today's streaming **ACT sub-steps only**:
  `_stream_provider_turn` + assistant-content emit + `_execute_tools_turn` + per-turn recovery.
- **What moves up into the shared phases:** the outer EVALUATE/DECIDE work `_stream_turn()` does
  inline today — `_evaluate_provider_turn_stops`, `_detect_task_completion_and_mentions`,
  `_evaluate_post_tool_turn`, `_handle_continuation_decision`, nudge emission — is **deleted from the
  streaming path** and provided once by `AgenticLoop._evaluate` / `_decide`. Streaming-specific I/O
  shrinks to: (a) ACT yields chunks, (b) DECIDE's terminal/nudge result is rendered as a chunk.
- **`StreamingChatExecutor` shrinks** to the streaming-ACT primitive (provider SSE + emit + tools +
  per-turn recovery) exposed via the `execute_turn_streaming` port; its outer `while`-driver is
  removed. `ChatStreamRuntime.get_executor()` is repointed to drive `AgenticLoop.run_streaming()`,
  and `ChatService.stream_chat()` consumes that iterator unchanged.
- **Public contracts unchanged:** `Agent.run()` → `CompletionResponse` and `Agent.stream()` →
  `AgentExecutionEvent`/`StreamChunk` are both preserved; only the *internal* driver behind
  `stream_chat()` changes.

### Capabilities the UI path gains (the "feature-complete" payoff)

Per-turn, the streaming path inherits the headless loop's research-rooted mechanisms:

- **Per-turn PERCEIVE** with calibrated confidence (truth-aligned uncertainty, arXiv:2604.00445).
- **PLAN + fast-slow planning gate** (arXiv:2604.01681) — skips/!skips LLM planning per turn.
- **Structured DECIDE** with enhanced-completion evaluation + fulfillment criteria.
- **Adaptive iteration / plateau termination** (intermediate-reward progress, arXiv:2604.07415).
- **Semantic response cache** (arXiv:2508.07675) and **paradigm routing** (arXiv:2604.06753).

### Implementation plan (small, landable, single canonical path — no flag)

Consistent with the FEP's no-flag rule, each step replaces behavior in place behind the
behavioral-parity gate; there is no `run_streaming` feature toggle and no retained legacy streaming
driver once cutover lands.

1. **Define the streaming-ACT port.** Add `execute_turn_streaming(...)` to the turn-executor
   protocol; implement it by lifting the ACT-only sub-steps out of `_stream_turn()` (provider +
   emit + tools + per-turn recovery), returning chunks + a `TurnResult`. No outer-loop change yet;
   `StreamingChatExecutor.run()` keeps driving it. Gate: characterization battery stays green.
2. **Add `AgenticLoop.run_streaming()`.** Outer async-gen reusing the existing non-yielding
   `_analyze_turn`/`_plan`/`_evaluate`/`_decide`, calling the new ACT port. Not yet wired to the
   UI. Unit-test it against scripted turn-executors.
3. **Cut the UI over.** Repoint `ChatStreamRuntime.get_executor()` to `AgenticLoop.run_streaming()`;
   delete `StreamingChatExecutor.run()`'s outer `while`-driver and its now-duplicated inline
   EVALUATE/DECIDE. This is the behavior-changing step — gated by the new parity battery + live zai.
4. **Reconcile governance & recovery.** Keep the streaming REQUEST gate before the loop; render
   DECIDE/recovery terminals as `is_final` chunks (streaming cannot post-gate a fully-buffered
   response — preserve per-emit gating). Fold streaming recovery into the shared recovery seam.
5. **StateGraph executor.** `run_streaming` initially targets the in-class while executor only;
   streaming support for the `USE_STATEGRAPH_AGENTIC_LOOP` path is deferred (orthogonal).

### Risks & mitigations

- **Live-streaming hot path.** A regression breaks `victor ui`. Mitigate with the parity battery +
  live zai in **both** modes each increment; watch for GeneratorExit / cancel-scope tracebacks
  (the historical failure mode) since ACT is now an async-gen nested inside another async-gen.
- **Token-granularity emission.** Live token streaming must stay inside the provider sub-step
  (`_stream_provider_turn`); `run_streaming` re-yields, it does not buffer-then-dump.
- **Behavior shift is intended but must be principled.** New phases (semantic cache, planning gate)
  can legitimately change a scenario's tool sequence or short-circuit it. Any battery scenario whose
  expectation changes is updated **with a written justification**, never silently.
- **Governance asymmetry.** Streaming has a REQUEST gate but no RESPONSE post-gate (chunks already
  sent); the unified loop must not assume a buffered post-gate exists.

### Acceptance bar (revised for framing B)

- **NEW — run/stream behavioral parity battery:** `Agent.run(p)` and `Agent.stream(p)` produce
  equivalent final answer + tool sequence across the QA scenarios (S1/S2/M1/M2/C1/W1–W4/U1). This
  is the precise encoding of "streaming is I/O, not behavior" and is the primary gate. *(Does not
  exist yet — to be authored alongside step 2/3; agent-reported "parity" today is streaming-only
  characterization.)*
- The existing streaming characterization battery's **scenario-level invariants** (correct answer,
  bounded/expected tool sequence, no traceback) continue to hold; expectations that move due to the
  new phases are updated with justification.
- The headless `AgenticLoop` unit suites stay green; `TurnObservation` / `TurnEvaluationController`
  remain the one decision seam.
- Live zai verification in **both** `--headless` and UI streaming: same answers, no streaming
  tracebacks.
- Single canonical loop — no `run_streaming` flag, no retained legacy streaming driver.

### Out of scope / deferred

Streaming under the StateGraph executor (`USE_STATEGRAPH_AGENTIC_LOOP`); changes to the public
`StreamChunk` / `AgentExecutionEvent` types; any provider-level streaming changes.

## References

- PR #170 — shared `TurnEvaluationController` (prerequisite).
- PRs #176–#190 — Phase 2 per-turn helper extractions (ACT / EVALUATE / emit / recovery /
  continuation), plus #184 (streaming fulfillment fix).
- PR #192 — Phase 2 assembly: the single `_stream_turn()` primitive + thin `run()` driver.
- PRs #186/#188 — streaming characterization battery (`tests/integration/streaming/`);
  PR #193 — removal of the flag-gated comparison scaffolding after the no-flag decision.
- CLAUDE.md — Agentic Loop (Phase 10), Agent Runtime Target State.
