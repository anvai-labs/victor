---
fep: "0046"
title: "RL Outcome Measurement Integrity — outcome_category on rl_outcome"
type: Standards Track
status: Draft
created: 2026-11-15
modified: 2026-11-15
authors:
  - name: Vijaykumar Singh
    email: vijay@anvaiops.com
    github: vjsingh1984
reviewers: []
discussion: null
---

# FEP-0046: RL Outcome Measurement Integrity — outcome_category on rl_outcome

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

## Summary

Add an `outcome_category` column to the `rl_outcome` central fact table with two
values: `'task'` (the row's `success`/`quality_score` describe an observable
task-level effect — a tool ran, a workflow finished) and `'learner_internal'`
(the row describes a learner's internal check or decision — a grounding
threshold verdict, a patience check, a cache hit, a prompt-render bookkeeping
event). The category is resolved inside the single row-writer funnel
(`_insert_outcome_row`) from the producing event type, with an explicit
producer override and a deterministic learner-identity fallback, so every
direct `record_outcome` caller is covered without touching call sites. A
versioned v10 migration backfills seven months of history deterministically
(event-type metadata first, then learner identity, then the
`session_summary` pseudo-bucket marker), backed by a coordinator-side lazy
guard because unversioned-modern stores are stamped at the current version
without replaying steps. Task-level aggregations (system success metrics,
cross-vertical pattern mining) filter `outcome_category = 'task'`; per-learner
views deliberately keep every row so learner training and learner-level
analysis are unchanged.

## Motivation

Analysis of 191,688 RL outcomes (2026-04-24 → 2026-11-10, seven months) proved
that `rl_outcome.success` conflates two meanings depending on the producing
event. The measured symptom: task type `code_generation` reported **0% success**
across 110 outcomes — yet **every one of those rows** came from the
`grounding_threshold` (67) and `continuation_patience` (43) learners, whose
`success` column records whether their *internal check* passed, tagged with the
underlying task's `task_type`. Code generation was not failing; the ruler was
broken. The same asymmetric conflation pattern appears across learners
(`grounding_threshold` 0% on `analysis`, `continuation_patience` 100% on
`analysis`), 4.6% of all rows (8,913) are conflated, and any unfiltered
`GROUP BY task_type` query returns wrong numbers — poisoning dashboards,
cross-vertical pattern mining, and every downstream decision that reads these
metrics (provider routing, learner tuning). Full analysis:
[RL data summary](../docs/analysis/rl-data-summary-2026-11-15.md) and
[measurement-integrity spec](../docs/architecture/rl-measurement-integrity-spec.md).

A second instance of the same disease rode the `tool_selector` learner under
the pseudo task type `session_summary` (usage analytics aggregation artifacts),
and `metrics.py` per-learner queries filtered on a non-existent `learner_name`
column with exceptions swallowed — silently reporting zeros.

## Proposed Change

1. **Taxonomy** (`victor/core/rl/hooks.py`): `TASK_LEVEL_EVENTS` and
   `LEARNER_INTERNAL_EVENTS` frozensets partition `RLEventType` exactly
   (enforced by a unit test so every new event type forces a conscious
   classification). `classify_event_category()` maps an event type to the
   category. `MODE_TRANSITION` is `learner_internal`: its `success=True` is
   hardcoded at the emit site, so counting it as a task outcome inflates task
   success denominators with rows that measure nothing.
2. **Public API** (`victor/framework/rl/base.py`): `RLOutcome.outcome_category:
   Optional[str] = None`. `None` means unspecified and defers to insert-time
   resolution; producers that know their category set it explicitly
   (e.g. usage-analytics session-summary rows set `learner_internal`).
3. **Resolution chain** (`victor/framework/rl/coordinator.py`
   `_insert_outcome_row` — the single writer all three record paths funnel
   through): explicit `outcome.outcome_category` → `metadata['event_type']`
   mapped through the taxonomy → `LEARNER_INTERNAL_LEARNERS` identity fallback
   → `'task'`. The resolved value is written to the row and assigned back onto
   the outcome so learner, row and telemetry agree.
4. **Schema** (`victor/core/schema.py`): `outcome_category TEXT DEFAULT 'task'`
   in the `rl_outcome` CREATE TABLE; versioned v9→v10 migration (ALTER +
   deterministic backfill: `metadata.event_type` first — `tool_selector`
   legitimately holds both task and internal event rows — then `learner_id`,
   then remaining NULLs to `'task'`); shared SQL constants so the versioned
   steps and the coordinator's lazy guard (needed because unversioned-modern
   stores are stamped without replaying steps) stay byte-identical.
5. **Read sites**: system-wide success metrics and the three cross-vertical
   task-level aggregations filter `outcome_category = 'task'`; per-learner
   metrics, per-session reads, and the mode-probe query deliberately do not.
   Fixes the `learner_name`/`timestamp` column bugs in `metrics.py` and raises
   their silent `logger.debug` exception swallows to `logger.warning`.

`task_type` semantics are unchanged — learners that build context keys from
`outcome.task_type` (continuation_patience, continuation_prompts,
quality_weights) are untouched, which is why the spec's Option A was chosen.

## Benefits

- Task-level success metrics become trustworthy immediately: the deterministic
  backfill covers seven months of history, `code_generation` stops reporting a
  phantom 0%, and the system-wide success rate stops being inflated by tens of
  thousands of hardcoded-success bookkeeping rows (on the production database,
  85,032 of 191,688 rows are learner-internal; corrected task-level success is
  93.9% rather than the conflated 95.1%).
- The partition test makes classification a deliberate, reviewed decision for
  every future event type instead of an accident that silently re-pollutes
  metrics.
- Learner-internal signals remain fully available for learner-level analysis
  and training; only the conflation ends. Learners are untouched (verified:
  they key on `task_type`/metadata, never on the new column).
- Fixes two silent observability bugs that this work surfaced: per-learner
  metrics queried a non-existent `learner_name` column (actual `learner_id`)
  with exceptions swallowed at debug level, so they always reported zeros, and
  `active_learners` was always empty for the same reason.
- The corrected data reveals where task-type granularity is real versus an
  artifact of conflation — direct input for the follow-on reliability and
  routing specs, which were previously at risk of optimizing against a broken
  ruler.

## Drawbacks and Alternatives

- **Option B (learner-specific task_types)** from the spec: rejected — breaks
  learners that key on `task_type` (continuation_patience, continuation_prompts,
  quality_weights), overloads the field further, and loses the ability to query
  "all outcomes recorded during X tasks".
- **Option C (separate tables)**: rejected — major schema surgery, breaks
  existing consumers and tooling, unnecessary for the problem.
- **Fix-forward only (no backfill)**: rejected — SQLite's `ADD COLUMN ...
  DEFAULT` fills existing rows at read time with the default (never NULL), so
  without an explicit backfill every historical internal row would sit
  permanently in task-level metrics or force `IS NULL OR = 'task'` cruft into
  every query forever; either way corrected metrics would wait weeks for new
  data.
- Cost: one more column on the hottest table (no index; full scans are
  milliseconds at 191K rows), a one-time sub-second backfill inside the
  migration transaction, and an existence probe on coordinator init that scans
  until the first pending row (unmeasurable on a classified database).
- Acknowledged behavior deltas: cross-vertical sample counts shrink (intended
  — internal rows no longer count toward sufficiency gates), per-learner
  metrics go from silent zeros to real values (bug fix), and system success
  rates change to reflect only genuine task outcomes (the point of the FEP).

## Unresolved Questions

- Should the historical `session_summary` rows recorded under `tool_selector`
  be re-typed beyond the category column (e.g. their own learner id)? Deferred;
  the category flag removes them from task-level metrics without a data move.
- Should per-provider success dashboards split by category? Follow-up work in
  the provider-routing spec (P2), which depends on this fix.

## Implementation Plan

Implemented alongside this proposal (single PR):

1. Taxonomy constants + `classify_event_category` + dispatch passes the
   category (`victor/core/rl/hooks.py`).
2. `RLOutcome.outcome_category` + dict round-trip
   (`victor/framework/rl/base.py`).
3. Resolution chain + column in `_insert_outcome_row` + lazy
   `_migrate_add_outcome_category` guard
   (`victor/framework/rl/coordinator.py`).
4. CREATE TABLE column, `CURRENT_SCHEMA_VERSION = 10`, v10 steps, shared SQL
   constants (`victor/core/schema.py`).
5. Read-site filters + metrics column fixes (`victor/framework/rl/metrics.py`,
   `victor/framework/rl/learners/cross_vertical.py`).
6. Explicit category for session-summary rows
   (`victor/agent/usage_analytics.py`).

Tests: event-set partition enforcement; parametrized classification of all 17
event types; dispatch sets category while preserving `task_type`; precedence
tiers (explicit > event_type > learner default > task); v10 migration on a
hand-built v9 DB (grounding / tool_selected / model_selector / no-event /
malformed-JSON rows); migration rerun is a no-op; lazy guard on an unversioned
DB; system metrics exclude learner-internal rows; cross-vertical counts use
task rows only.

## Migration Path

The migration is a single versioned step set (schema v9 → v10) plus an
equivalent coordinator-side lazy guard:

1. **Versioned stores** (`schema_version < 10`): the executor runs, in order,
   `ALTER TABLE rl_outcome ADD COLUMN outcome_category TEXT DEFAULT 'task'`,
   the learner-internal backfill (matching on the default-filled value and the
   `IS NOT` null-safe inequality), and the residual NULL → `'task'` backfill,
   all inside one `BEGIN IMMEDIATE` transaction with the version published
   atomically. Measured on a copy of the 191,688-row production database:
   0.52s, idempotent on rerun, no index added.
2. **Unversioned-modern stores** (stamped at the current version without
   replaying steps): the coordinator's `_migrate_add_outcome_category` guard
   adds the same column (byte-identical DDL from shared schema constants) and
   runs the same backfills, guarded by an existence probe so a classified
   database does no writes. This also self-heals rows written by old code
   (which omit the column and read as `'task'`) on the next coordinator init.
3. **Rollout**: update and first launch with no other victor process running
   (the one-time backfill holds the global DB write lock; WAL keeps readers
   unblocked). Take a database backup before the first post-update launch;
   code rollback after migration is safe (additive column), database rollback
   requires the backup.

## Compatibility

| | old code | new code |
|---|---|---|
| **old DB** | unchanged behavior | versioned v10 migration (versioned stores) or lazy guard (unversioned-modern stores) add + backfill the column before first write |
| **new DB** | named-column INSERTs omit the column → `DEFAULT 'task'` preserves today's semantics | full behavior |

Code rollback after migration is safe (additive column). DB rollback requires
the pre-migration backup. Concurrent global-DB writers during the one-time
backfill: run the upgrade first-launch with no other victor process running.

## References

- [RL data summary 2026-11-15](../docs/analysis/rl-data-summary-2026-11-15.md)
- [Deep impact analysis](../docs/analysis/rl-specs-deep-impact-analysis.md)
- [Measurement integrity spec](../docs/architecture/rl-measurement-integrity-spec.md)
  (superseded by this FEP; its Open Questions 1–3 are resolved here: backfill
  yes, default `'task'`, monitoring as follow-up)
- [FEP-0033](fep-0033-rl-subsystem-relocation.md) — RL subsystem relocation
  (why the coordinator lives in `victor/framework/rl/`)
