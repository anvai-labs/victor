---
fep: "0043"
title: "Caller-Declared Completion Criteria Gate"
type: Standards Track
status: Draft
created: 2026-11-08
modified: 2026-11-09
authors:
  - name: Vijaykumar Singh
    email: vijay@anvaiops.com
    github: vjsingh1984
reviewers: []
discussion: null
---

# FEP-0043: Caller-Declared Completion Criteria Gate

## Table of Contents

1. [Summary](#summary)
2. [Motivation](#motivation)
3. [Proposed Change](#proposed-change)
4. [Benefits](#benefits)
5. [Drawbacks and Alternatives](#drawbacks-and-alternatives)
6. [Unresolved Questions](#unresolved-questions)
7. [Implementation Plan](#implementation-plan)
8. [Compatibility](#compatibility)
9. [References](#references)

## Summary

Add an opt-in post-filter on the EVALUATE seam — the `criteria_gate` — that
downgrades a COMPLETE decision to RETRY while caller-declared completion
minimums (declared via `run(query, context={"criteria": {...}})`) remain
unmet and iterations remain. Victor stays a generic framework: criteria are
opaque caller key/values evaluated against the existing generic fulfillment
strategy vocabulary (`min_findings`, `min_results`, ...); the framework
never learns what a criterion means. The same gate also honors a
caller-declared completion marker (`complete_on_key`): a successful tool
result carrying that state key with a truthy value records the marker, and
an enabled gate upgrades a non-COMPLETE decision to COMPLETE with a
deterministic reason — for consumers whose tools declare completion
directly (e.g. a proposal admission that closes a corpus) instead of by
countable collection.

The motivation is a measured consumer failure: a 32-iteration funded
research run with a declared coverage objective of 30+ findings
self-completed at iteration 12 with 13 findings — the fulfillment
evaluation was advisory (early-exit only, never blocking COMPLETE), the
task classifier and topology selector silently clamped the funded
iteration budget, and the model's own completion narrative satisfied the
generic cascade. The caller had no first-class way to declare "this
objective is a floor, not a suggestion". This FEP provides three generic
mechanisms: the binding criteria gate, budget-clamp surfacing (an
explicit caller iteration budget binds over the task-classifier
heuristic), and stage-scoped guidance overrides — each opt-in,
default-off, and usable by any agentic consumer.

## Motivation

A consumer funded a 32-iteration research run with a declared coverage
objective of 30+ findings. The loop self-completed at iteration 12 with 13
findings: the fulfillment evaluation is advisory (it can trigger an early
exit but never blocks COMPLETE), the task classifier and topology selector
silently clamped the funded iteration budget, and the model's own
completion narrative ("I have enough context") satisfied the generic
completion cascade. The caller had no first-class way to say "this
objective is a floor, not a suggestion".

Two existing mechanisms partially cover this and were rejected:
`AnalysisFulfillment.check` already implements `min_findings` semantics,
but the fulfillment result is advisory; `ModeCompletionConfig
.check_early_exit` refuses early exit but cannot block a COMPLETE emitted
from any of the ~10 completion-cascade return points. The `effect_gate`
post-filter seam (ADR-010/EVR-4) is the proven place that gates every
cascade return.

## Proposed Change

1. `CriteriaGate` (new `victor/framework/criteria_gate.py`): an opt-in
   post-filter mirroring `EffectGate` conventions. On a COMPLETE
   evaluation, read `state["criteria"]` (seeded from
   `run(query, context={"criteria": {...}})` — `state` already merges
   context at `AgenticLoop.run`). A criterion key maps to a minimum
   count; the gate counts matching state entries (list/dict lengths) and
   downgrades COMPLETE to RETRY with a deterministic reason enumerating
   every unmet minimum.
2. `AgenticLoopConfig.enable_criteria_gate: bool = False` and the gate
   constructed/reset/applied alongside `effect_gate` on the same seam.
3. Budget-clamp surfacing: an explicit caller `iteration_budget`
   override binds over the task-classifier budget (the classifier's
   `tool_budget * 2` heuristic still applies when no override is given),
   and `PARALLEL_EXPLORATION` grounding preserves a declared override
   instead of clamping to 2.
4. Stage-scoped guidance override: `OperationalModeConfig` gains
   optional `stage_prompt_overrides` so a consumer can replace mode
   guidance lines (e.g. the BUILD-mode "STOP excessive reading" line)
   for its own stage key without mutating `MODE_CONFIGS` globally.
5. Generic state mirroring (`criteria_gate.mirror_criteria_state`): the
   gate counts state entries; this seam lets a run's own tool evidence
   populate them. At the ACT/record seam (buffered `_act` and the
   streaming post-ACT block), for every criteria-declared key, each
   successful tool result carrying a same-named list contributes its
   unique entries (a corrected re-proposal that re-reports prior
   entries counts once — a floor must never be met by double
   counting) and a same-named dict merges by key. The framework never
   interprets what a declared key collects; consumers declare the key
   and shape their tool payloads. Opt-in by declaration and a no-op
   without criteria; the gate itself stays a pure decision post-filter.
6. Caller-declared completion marker: `run(query,
   context={"complete_on_key": "..."})` names one state key the
   consumer's own tools emit. `mirror_criteria_state` records the marker
   when a successful tool result carries that same-named key with a
   truthy value of any JSON type (failed results and falsy values never
   record it), and `CriteriaGate.apply` upgrades a non-COMPLETE decision
   to COMPLETE with the deterministic reason "Caller completion marker
   present (state key 'X')" plus `criteria_gate.completed_by_marker`
   metadata. This covers consumers whose tools model completion as a
   state transition ("the proposal is admitted; source access closes")
   rather than a countable collection. The upgrade never fires with the
   gate disabled, without the declaration, or while the marker is absent
   (default-off parity, ADR-012), and it never rewrites a COMPLETE — the
   criteria downgrade/annotate paths own those. Measured motivation: a
   research consumer's loop always ran one more model turn after its own
   tool admitted a proposal and closed the corpus, and that ack turn's
   request carried the full transcript past the whole-run input-byte cap,
   killing a run that held a valid admitted proposal.

## Benefits

- Callers can fund deep runs with declared floors and get them: the
  measured consumer run moved from 13 to 24 findings on prompt pressure
  alone; a binding gate completes the mechanism.
- Generic: criteria are opaque caller key/values against the existing
  fulfillment vocabulary — no domain leakage into the framework.
- Default-off parity (ADR-012): identical behavior when disabled.
- Bounded: the gate downgrades at most `max_downgrades` times before
  COMPLETE stands annotated, preventing unbounded RETRY cycles and
  plateau-detector false FAILs on funded deep runs.
- Budget-clamp surfacing: an explicit caller iteration budget reaches
  the loop instead of being silently halved by the classifier heuristic.
- Stage-scoped guidance: consumers replace mode guidance lines for their
  own stage key without touching the global mode registry.
- Completion-marker consumers stop at their own tool's completion
  signal: no extra model turn after the run's work is done, so funded
  whole-run input budgets are not spent re-sending a finished transcript.

## Drawbacks and Alternatives

- False blocks: a legitimately finished task can be held to RETRY.
  Bounded: the gate downgrades at most ``max_downgrades`` (default 2,
  AgenticLoopConfig-threaded) times, after which COMPLETE stands
  annotated with the unmet minimums (best-effort retention); it also
  yields COMPLETE when the run context marks the iteration budget
  exhausted. Criteria minimums are caller-chosen; default-off.
- Accepted consequence: with an explicit caller iteration_budget in the
  run context, the PARALLEL_EXPLORATION clamp-2 heuristic is bypassed for
  ALL such runs (chat_stream_helpers always populates iteration_budget
  from stream settings), not only for callers that declare criteria. The
  un-clamp is the FEP's stated intent — a caller-funded budget binds over
  topology heuristics; if "explicit" ever needs to distinguish
  prompt-declared budgets from settings-derived ones, a separate marker
  key in the run context is the follow-up.
- Rejected alternative: extending `ModeCompletionConfig
  .check_early_exit` — it governs early exit only and cannot block
  COMPLETE from the ~10 completion-cascade return points.
- Rejected alternative: making the advisory fulfillment result binding —
  it would change every task type at once; the opt-in gate scopes the
  behavior change to callers that declare criteria.

## Unresolved Questions

- Should unmet-criteria CONTINUE reasons feed the spin detector with a
  distinct signature? (Current design: deterministic reason + unmet
  list; observed no false spin trips in the reference consumer.)

## Implementation Plan

1. `victor/framework/criteria_gate.py`: `CriteriaGateConfig`,
   `CriteriaGate` with `unmet_criteria(state)` and `apply(...)`.
2. `AgenticLoopConfig.enable_criteria_gate`; construction/reset/apply on
   the EVALUATE seam after `per_turn_auditor`.
3. Budget-clamp surfacing in `turn_execution_runtime` and
   `topology_selector` (explicit caller override binds over the
   classifier heuristic; executor ceiling stays).
4. `stage_prompt_overrides` on `OperationalModeConfig` with a generic
   registration/consumption API.
5. Enablement: the gate is framework-level (`AgenticLoopConfig
  .enable_criteria_gate`, default off). Production chat/buffered executors
  thread it in a follow-up; direct AgenticLoop callers (reporeel's
  research worker) can enable it immediately via `UnifiedAgentConfig
  .extra`. `criteria_gate_max_downgrades` is threaded from
  `AgenticLoopConfig` alongside the flag (int-coerced/clamped >= 0 at
  use, mirroring EffectGate).
6. SDK run-context forwarding: the buffered message seam
   (`execute_message` -> `ChatService.chat` -> `execute_agentic_loop`)
   forwards the structured caller `context` as `run_context` alongside
   the flattened prompt text, with compatibility-aware keyword guards so
   runtimes predating the keyword are untouched. The loop's context
   merge places internal seeds last, so reserved keys (task
   classification, budgets, overrides) can never be clobbered by caller
   keys. Streaming (`Agent.stream` -> `stream_message_events`) forwards
   the flattened text only today; structured streaming forwarding is a
   documented follow-up — buffered is the path consumers gate on.
7. State mirroring wiring: the buffered `_act` seam and the streaming
   post-ACT block call `mirror_criteria_state` after tool results are
   recorded, so the gate counts the run's own evidence (opt-in by
   declaration, no-op without criteria). The same wiring records the
   `complete_on_key` marker; the gate's apply path upgrades on it with no
   new loop wiring.

## Migration Path

None required. The gate is opt-in (`enable_criteria_gate`, default
off), criteria are declared per-run via the context, and every default-
off decision path is byte-identical to current behavior. Consumers on
older victor versions see advisory fulfillment exactly as today.
Retained artifacts and replay contracts are unaffected: the gate reads
run-time state only, and prompt digests are pinned per job by callers.

## Compatibility

Default-off parity per ADR-012: with the gate disabled or no criteria
or completion marker declared, every decision path is byte-identical to
current behavior. The `complete_on_key` upgrade requires three
independent conditions (gate enabled, declaration present, marker
recorded from a successful tool result); any one missing keeps the
evaluation decision untouched. No schema, storage, or wire changes.
Consumers on older victor versions see advisory fulfillment exactly as
today.

## References

- `EffectGate` (ADR-010 / EVR-4): the post-filter seam convention.
- `AnalysisFulfillment.check` (`fulfillment.py:607-613`): the generic
  `min_findings` vocabulary.
- `ModeCompletionConfig.check_early_exit`: the insufficient
  early-exit-only mechanism this FEP generalizes.
