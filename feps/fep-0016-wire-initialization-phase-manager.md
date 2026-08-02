---
fep: "0016"
title: "Wire the initialization phase manager (centralize orchestrator runtime init)"
type: Standards Track
status: Implemented
created: 2026-07-10
modified: 2026-07-10
authors:
  - name: Vijaykumar Singh
    email: vijay@anvaiops.com
    github: vjsingh1984
reviewers: []
discussion: https://github.com/anvai-labs/victor/discussions/0016
---

# FEP-0016: Wire the initialization phase manager

## Table of Contents

1. [Summary](#summary)
2. [Motivation](#motivation)
3. [Proposed Change](#proposed-change)
4. [Benefits](#benefits)
5. [Drawbacks and Alternatives](#drawbacks-and-alternatives)
6. [Unresolved Questions](#unresolved-questions)
7. [Implementation Plan](#implementation-plan)
8. [Migration Path](#migration-path)
9. [Compatibility](#compatibility)
10. [References](#references)
11. [Review Process](#review-process)
12. [Acceptance Criteria](#acceptance-criteria)

---

## Summary

`AgentOrchestrator` is documented as a facade, but its runtime initialization is
spread as **9 `_initialize_*` phases invoked by raw direct calls scattered across
three files** (`orchestrator.__init__`, `component_assembler`, `bootstrapper`). A
purpose-built centralizer, `InitializationPhaseManager` (`victor/agent/runtime/
initialization_manager.py`), already exists — it declares the phase order,
criticality (fail-fast), dependency-skipping, timing, and structured error
reporting — **but it is dead code: zero importers, zero callers**. Because the
manager was the intended home, one phase (`credit_runtime`) was registered *only*
on it and so never ran in production until FEP-0016's precursor fix (PR #464).

This FEP makes the manager real: it becomes the **single source of truth for the
init-phase contract** (order, criticality, dependencies, timing, errors), and the
orchestrator invokes it instead of making raw scattered `_initialize_*` calls.

Crucially — and unlike the manager's current `run_all_phases(self)` docstring
implies — the phases **cannot** collapse to a single call: they are structurally
interleaved with component assembly (phases 5–6 need `_context_compactor`, built
mid-`assemble_intelligence`; phases 7–9 need `_checkpoint_manager`, built in
`prepare_components`). This FEP therefore proposes invoking the manager at the
**natural lifecycle boundaries** (grouped phase runs), not a single call. Public
API is unchanged; this restructures an internal core-architectural pattern
(init sequencing), which is why it requires an FEP.

## Motivation

### Problem Statement

Verified (grep + read, 2026-07; backlog **F-002**):

- `victor/agent/orchestrator.py` is ~4,700 LOC and documented (docstring ~L438-445)
  as a facade that delegates "without containing business logic".
- **9 `_initialize_*` phases**, invoked by raw direct calls at 3 sites:
  - `orchestrator.__init__`: `_initialize_provider_runtime` (~L946), `_initialize_metrics_runtime` (~L1066), `_initialize_workflow_runtime` (~L1084), `_initialize_memory_runtime` (~L1096)
  - `component_assembler.assemble_intelligence`: `_initialize_resilience_runtime(context_compactor=…)` (L389), `_initialize_coordination_runtime` (L392)
  - `bootstrapper.prepare_components`: `_initialize_interaction_runtime` (L185), `_initialize_services` (L188), `_initialize_credit_runtime` (added by #464)
- **`InitializationPhaseManager` is dead code**: `grep` finds no import of the module
  and no caller of `run_all_phases` anywhere in the tree (the one `manager = …` occurrence
  is inside its own class docstring). It already encodes the phase order, `critical`
  fail-fast flags, dependency-based skipping, per-phase timing, and structured
  `InitializationResult` — all currently unused.
- **Consequence already observed**: `credit_runtime` was registered *only* on the dead
  manager, so it never ran in production; enabling `settings.credit_assignment.enabled`
  silently did nothing. Fixed as a precursor in **PR #464** by adding the missing raw
  call — which this FEP will subsume into the manager.

### The interleaving constraint (the crux — verified file:line)

A single `run_all_phases(self)` at one point is **infeasible** because init phases
depend on components built *between* them:

- `_context_compactor` is created at `component_assembler.py:299` (inside
  `assemble_intelligence`, which itself needs the conversation/memory built by
  earlier assembly). Phases 5–6 (`resilience`, `coordination`) run at
  `component_assembler.py:389,392` and **require** it (passed as an argument, L390).
- `_checkpoint_manager` is created at `bootstrapper.py:152` (inside
  `prepare_components`). Phase 7 (`interaction`) runs at `bootstrapper.py:185` and
  requires it, plus the assembled `tool_pipeline`/`conversation_controller`/etc.

So the natural structure is three phase-groups at three lifecycle boundaries:
**Group A** phases 1–4 (early, in `__init__`), **Group B** phases 5–6 (after
`_context_compactor`, in `assemble_intelligence`), **Group C** phases 7–9 (after
`_checkpoint_manager` + assembly, in `prepare_components`).

### Goals

1. Make `InitializationPhaseManager` the **single source of truth** for the init-phase
   contract (order, criticality, dependency-skip, timing, structured errors), and
   actually invoke it — no more raw scattered `_initialize_*` calls.
2. Preserve behavior exactly (same phases, same order, same effects) while gaining
   fail-fast on critical phases + structured `InitializationResult` + per-phase timing.
3. Fold `credit_runtime` into the manager path (removing the #464 stopgap call).
4. Correct the manager's model to reflect the **real interleaving** (grouped runs at
   3 boundaries), not the fictional single call.

### Non-Goals

- Changing the orchestrator's public constructor signature or any public API.
- Extracting the *bodies* of the `_initialize_*` methods into services (a separate,
  later concern; this FEP relocates *invocation*, not implementation).
- Reordering component assembly to force fewer call points (documented as Alternative B,
  explicitly deferred as higher-risk).

## Proposed Change

### Design A (recommended): manager owns the contract, invoked at 3 boundaries

Give `InitializationPhaseManager` grouped runners that execute a contiguous slice of
its declared phases with the same structured handling it already has:

```python
# victor/agent/runtime/initialization_manager.py
class InitializationPhaseManager:
    def run_group(self, orchestrator, group: PhaseGroup) -> InitializationResult:
        """Run the phases in `group` in declared order, with per-phase timing,
        dependency-skip, and fail-fast on `critical` phases. Returns the result."""
```

Replace the raw scattered calls with grouped manager calls at the exact same points:

```python
# orchestrator.__init__ (was: 4 raw _initialize_* calls)
self._init_manager = InitializationPhaseManager()
self._init_manager.run_group(self, PhaseGroup.EARLY)     # provider, metrics, workflow, memory

# component_assembler.assemble_intelligence (was: resilience + coordination raw)
orchestrator._init_manager.run_group(orchestrator, PhaseGroup.ASSEMBLY)  # after _context_compactor

# bootstrapper.prepare_components (was: interaction + services + credit raw)
orchestrator._init_manager.run_group(orchestrator, PhaseGroup.SERVICE)   # after _checkpoint_manager
```

The manager remains the one place that declares phase order, which phases are
`critical` (fail-fast), dependency-skip rules, and produces a per-group
`InitializationResult` the caller checks. The `_initialize_*` method bodies are
unchanged. The credit phase moves into `PhaseGroup.SERVICE`, removing #464's raw call.

### Detailed Specification

- **Phase grouping** is declared once in the manager (EARLY = 1–4, ASSEMBLY = 5–6,
  SERVICE = 7–9), matching the verified interleaving. The `context_compactor` argument
  for the resilience phase is read from `orchestrator._context_compactor` at call time
  (it exists by the ASSEMBLY boundary) rather than threaded as a parameter.
- **Fail-fast**: `provider_runtime` and `interaction_runtime` stay `critical`; a failed
  critical phase raises `InitializationError` (today a raw failure just propagates —
  behavior-compatible, but now structured).
- **Result handling**: each `run_group` returns `InitializationResult`; the caller logs
  failed/skipped phases. Non-critical phase failure is logged and skipped exactly as the
  manager already specifies.
- **No behavior change**: the phases, their order, and their side-effects are identical;
  only the *invocation mechanism* changes (raw call → manager-driven, with added timing
  and structured errors).

### API Changes

Internal only. `InitializationPhaseManager` gains `run_group()` (and `PhaseGroup`).
No `victor/framework/` or public `AgentOrchestrator` API changes.

## Benefits

### For Framework Maintainers
- One authoritative init-phase contract (order/criticality/skip/timing) instead of
  knowledge spread across three files; adding/reordering a phase is a one-place edit.
- Structured `InitializationResult` + per-phase timing → real observability into a
  currently opaque, latency-relevant construction path.
- Fail-fast on critical phases with a clear `InitializationError` instead of an
  arbitrary deep exception.

### For the Ecosystem
- Kills a 240-LOC dead abstraction that already misled once (the `credit_runtime`
  bug) by making it live and correct.

## Drawbacks and Alternatives

### Drawbacks
- Touches the orchestrator construction hot path (high blast radius). *Mitigation*:
  behavior-preserving; phased rollout group-by-group; each group validated against the
  full orchestrator-construction test set before the next.
- The manager's declared order must be corrected to the real interleaving (its current
  docstring implies a single call). *Mitigation*: the grouping is part of this FEP and
  guard-tested.

### Alternatives Considered

1. **Alternative B — reorder construction for a single (or 2-point) call.** Front-load
   `_checkpoint_manager` before assembly so phases 7–9 run right after
   `assemble_intelligence`, collapsing to ~2 call points. *Rejected for now*: reordering
   `_checkpoint_manager` and ensuring `prepare_components`' other setup
   (`wire_component_dependencies`, `vertical_context`) doesn't sit between checkpoint and
   phase 7 is materially riskier on a hot path for a marginal call-count reduction.
   Revisit after Design A lands.
2. **Delete the manager as dead code.** ~240 LOC removed, keep the scattered calls as the
   honest reality. *Rejected*: loses the fail-fast/timing/structured-error value the
   manager already implements; the user elected to wire it.
3. **Leave as-is.** *Rejected*: perpetuates the scattered contract and the class of bug
   #464 fixed (a phase added only to the dead manager silently never runs).

## Unresolved Questions

- **Group vs. per-phase API**: expose `run_group(group)` (3 calls) or keep 3 explicit
  group methods (`run_early`/`run_assembly`/`run_service`)? (Proposed: a single
  `run_group(PhaseGroup)` for one code path.)
- **Should ASSEMBLY/SERVICE groups also become `critical`-aware at the group level**, or
  keep per-phase criticality only? (Proposed: per-phase, unchanged.)
- Does any non-init step currently run *between* two phases *within* a group (would break
  grouping)? Phase-1-scoped verification will confirm before wiring each group.

## Implementation Plan

### Phase 1: Manager readiness + grouping (1 PR)
- [ ] Add `PhaseGroup` + `run_group()` to `InitializationPhaseManager`; declare EARLY/ASSEMBLY/SERVICE membership matching the verified interleaving.
- [ ] Unit-test the manager directly (order, fail-fast, skip, timing) — it currently has tests that mock the orchestrator; extend them for `run_group`.
- [ ] No orchestrator changes yet. **Deliverable**: manager is invocable and tested; still not wired.

### Phase 2 (DONE, PR #477) — **revised from grouped to per-phase in-place**

Implementation revealed the grouped design was infeasible: (a) the 4 EARLY phases are
**scattered across ~120 lines of `__init__`** with real construction (sanitizer, prompt
builder, prompt pipeline, runtime-intelligence) between them — they can't consolidate into
one `run_group(EARLY)` without proving/moving those prerequisites; and (b) incremental
grouped wiring hits a **cross-group dependency-accumulation** problem (wiring SERVICE first
would skip its critical `interaction` phase because `provider`/`coordination` ran raw).
Approved pivot to **per-phase in-place wiring**:
- [x] Add `run_phase(orchestrator, name)` to the manager.
- [x] Create `self._init_manager` once in `__init__`; replace each of the 9 raw
  `_initialize_*` calls with `run_phase("X")` **at its existing site** (nothing moves →
  behavior-preserving; the manager accumulates succeeded-phase state across the calls in
  construction order, so dependencies resolve).
- [x] Guard test: no raw `_initialize_*` call sites remain in the construction path; the
  contract covers all 9 phases.
- [x] Behavior parity verified (clean develop and the branch show the identical 44
  pre-existing async-env local failures; CI ran the async tests properly and passed).
  CI additionally caught two real gaps (a `SimpleNamespace` test orchestrator missing
  `_init_manager`; `orchestrator.py` briefly 9 lines over its hotspot cap) — both fixed.

### Phase 3 (optional follow-up)
- [ ] Prune the now-unused `run_group` + `PhaseGroup` + `_PhaseSpec.group` (built in Phase 1
  for the abandoned grouped design; `run_phase` is the production path). Low value — the
  manager is now actively used, so these are stray methods, not a dead subsystem.

### Testing Strategy
- Each wiring PR runs the full `tests/unit/agent/` orchestrator-construction set + a
  smoke test that constructs a real orchestrator and asserts all 9 phases ran (via the
  `InitializationResult`), before and after.
- A behavior-parity check: the set of orchestrator attributes populated after
  construction is identical pre/post each PR.

## Migration Path

Internal refactor; no external migration. Each phase is independently revertible.

## Compatibility

- **Breaking change**: No (public API unchanged; behavior preserved).
- **Vertical compatibility**: none affected (verticals don't touch orchestrator init).
- **Deprecation**: n/a.

## References

- Backlog: `docs/architecture/CODEBASE_REVIEW_BACKLOG.md` → **F-002**.
- Precursor fix: PR #464 (`credit_runtime` wired into production; subsumed here).
- `victor/agent/runtime/initialization_manager.py` (the dead manager).
- Verified interleaving: `component_assembler.py:299,389,392`; `bootstrapper.py:152,185,188`.

## Review Process

### Submission
- **Submitted by**: Vijaykumar Singh
- **Date**: 2026-07-10
- **Pull Request**: TBD

### Review Checklist
#### Technical Review
- [ ] Phase grouping matches the real interleaving (no non-init step wedged within a group)
- [ ] Fail-fast/criticality semantics preserved
- [ ] Behavior-parity test approach adequate for a hot path

### Decisions
- **Recommendation**: Accept (**Design A**)
- **Decision date**: 2026-07-10
- **Approved by**: Vijaykumar Singh (repo owner)
- **Resolved choices**:
  - Approach: **Design A** — manager owns the phase contract, invoked at the 3 natural boundaries (grouped). Alternative B (reorder for fewer calls) rejected as worst risk/reward on the hot path; Alternative C (delete the manager) rejected in favor of the fail-fast/timing value.
  - API: single **`run_group(PhaseGroup)`** (one code path), not three named methods.
  - Criticality: **per-phase** `critical` flags unchanged (no group-level fail-fast).
  - Rollout: **SERVICE → EARLY → ASSEMBLY**, one PR per group, each behavior-parity validated and independently revertible.
- **Rationale**: The concrete headline bug (`credit_runtime` never running) is already fixed (#464); the remaining value is incremental but real — a single authoritative phase contract (so a phase can't be silently lost again), fail-fast on critical phases, and per-phase timing on an opaque hot path. Design A captures that at contained, behavior-preserving risk.

## Acceptance Criteria

### Must-Have Criteria
1. **Manager is the sole init-phase invoker**: no raw `orchestrator._initialize_*(` call
   sites remain outside `InitializationPhaseManager` (guard test).
2. **Behavior parity**: post-construction orchestrator state identical to pre-refactor;
   all orchestrator-construction tests green.
3. **All 9 phases run in production** (incl. `credit_runtime`), verifiable via
   `InitializationResult`.

### Should-Have Criteria
1. Per-phase timing surfaced in debug logs — Priority: Medium.

### Implementation Requirements
- [ ] Guard test (no raw init calls outside the manager)
- [ ] Behavior-parity smoke test
- [ ] Changelog entry; backlog F-002 → DONE on completion

---

## Copyright

This FEP is licensed under the Apache License 2.0, same as the Victor project.
