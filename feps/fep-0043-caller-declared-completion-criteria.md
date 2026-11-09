---
fep: "0043"
title: "Caller-Declared Completion Criteria Gate"
type: Standards Track
status: Draft
created: 2026-11-08
modified: 2026-11-08
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
never learns what a criterion means.

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

## Benefits

- Callers can fund deep runs with declared floors and get them: the
  measured consumer run moved from 13 to 24 findings on prompt pressure
  alone; a binding gate completes the mechanism.
- Generic: criteria are opaque caller key/values against the existing
  fulfillment vocabulary — no domain leakage into the framework.
- Default-off parity (ADR-012): identical behavior when disabled.

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

## Compatibility

Default-off parity per ADR-012: with the gate disabled or no criteria
declared, every decision path is byte-identical to current behavior.
No schema, storage, or wire changes. Consumers on older victor versions
see advisory fulfillment exactly as today.

## References

- `EffectGate` (ADR-010 / EVR-4): the post-filter seam convention.
- `AnalysisFulfillment.check` (`fulfillment.py:607-613`): the generic
  `min_findings` vocabulary.
- `ModeCompletionConfig.check_early_exit`: the insufficient
  early-exit-only mechanism this FEP generalizes.
