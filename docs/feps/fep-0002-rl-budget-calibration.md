---
fep: "0002"
title: "RL-Driven Tool-Budget Calibration"
type: Standards Track
status: Draft
created: 2026-07-08
modified: 2026-07-08
authors:
  - name: "Vijaykumar Singh"
reviewers: []
discussion: null
implementation: null
---

# FEP-0002: RL-Driven Tool-Budget Calibration

> **Legacy proposal series.** This document retains the old `docs/feps` number 0002
> as historical context. It is distinct from canonical
> [FEP-0002](https://github.com/anvai-labs/victor/blob/develop/feps/fep-0002-documentation-style-guide.md) in the root `feps/` series.
> Its Draft metadata is preserved. See [ADR-017](https://github.com/anvai-labs/victor/blob/develop/docs/architecture/adr/017-rl-budget-calibration.md)
> for the related current proposal or decision, and the [canonical FEP index](https://github.com/anvai-labs/victor/blob/develop/feps/README.md).


## Abstract

Introduce an opt-in, settings-gated pipeline that calibrates the session tool
budget (`tool_call_budget`) from aggregated reinforcement-learning signals
(`rl_tool_q` Q-values and `decision_outcome` aggregates) already stored in the
global Victor database. The pipeline is pure and three-phase — a signal reader,
a calibration policy, and an application seam — and applies an *immutable*
`ToolSettings` overlay only when (a) the user opts in and (b) the calibrator's
confidence exceeds a configurable threshold. The default-off path returns the
baseline unchanged, yielding zero behavior change until explicitly enabled. This
proposal covers the runtime wiring of the already-implemented (but unwired)
calibration modules, which is gated behind this FEP's review window per the
framework's public-API / runtime-behavior change policy.

## Motivation

The static `tool_call_budget` does not adapt to observed execution reality.
Aggregate analysis of the global RL database showed two principled signals the
static budget ignores:

- **Per-tool learned value** (`rl_tool_q`): high-selection, high-Q tools deserve
  more headroom; low-Q tools are budget sinks.
- **Outcome history** (`decision_outcome`): a high observed failure rate means
  budget is being spent on fruitless calls and should tighten; consistent
  success justifies relaxing.

Today the budget over-allocates when execution is failing and under-allocates
when high-value tools consistently succeed. Calibration closes that gap
principled-ly, reducing wasted tool calls / cost while preserving headroom for
productive sessions.

## Specification

### Pipeline (pure, three-phase)

1. **`BudgetSignalReader`** — loads aggregated `rl_tool_q` Q-values and
   `decision_outcome` success/failure/reward from the **existing** global DB
   (`~/.victor/victor.db`) via the existing `db.cursor()` / `Tables.*` pattern.
   No new tables, no new DB path.
2. **`BudgetCalibrator`** — pure policy producing a `BudgetRecommendation`
   (`recommended_tool_call_budget`, `confidence` in [0,1], `rationale`).
3. **`apply_budget_calibration()` seam** — the single composition entry point,
   gated by:
   - `tool_budget_calibration_enabled: bool = False`
   - `tool_budget_calibration_min_confidence: float = 0.7` (0.0-1.0; see Open
     Questions #1 — raised from 0.5 per FEP review)

### The identity-return contract

The seam returns `settings.tools` **by identity** when: the toggle is off,
confidence < threshold, confidence <= 0.0 (cold-start / degraded reads), or a
pipeline exception occurs. Consequences:

- **Zero behavior change by default** — default-off + identity-return means the
  feature is inert until explicit opt-in *and* sufficient confidence.
- **One wiring site, zero consumption-site changes** — the overlay is a new
  immutable `ToolSettings` with identical schema, so all ~30
  `settings.tools.tool_call_budget` readers consume the calibrated value with no
  edit.

### Wiring (this FEP's subject)

A single call inside `Agent.create()` in `victor/framework/agent.py`, immediately
after the existing `session_config.apply_to_settings(settings)` call (currently
at line 241):

    from victor.framework.rl.budget_calibration_seam import apply_budget_calibration
    settings.tools = apply_budget_calibration(settings)

(Review correction: the original draft cited `AgentFactory.create()`; that
method contains only `_apply_profile_overrides()` / `validate()` /
`ensure_bootstrapped()` and has no SessionConfig application point. The real
apply point is `Agent.create()`.)

Because the seam is identity-returning when disabled, this line is a no-op unless
a user opts in.

**Precedence (review decision #2):** an explicit session/CLI budget override
must win absolutely. The seam accepts an `explicit_override: bool` flag and
short-circuits to identity-return when it is `True` (e.g. when
`session_config.tool_budget` was provided). NOTE: `SessionConfig.tool_budget` is
currently validated but *not* applied inside `apply_to_settings()` (see
Open Questions #2 / Additional Risks) — the precedence wiring is blocked on a
separate fix that actually applies that override.

### UX and observability (non-runtime, not FEP-blocked)

- Structured logging at the seam on every apply decision, with confidence,
  baseline, recommended budget, and rationale. This is the *only* place runtime
  provenance surfaces; it is not threaded through `ToolSettings` (config stays
  pure).
- `/system` surfaces *configured* calibration state — the
  `tool_budget_calibration_enabled` flag (on/off) and the
  `tool_budget_calibration_min_confidence` value. It does **not** show runtime
  provenance (per-call confidence/baseline/rationale), because the seam returns
  only the resulting `ToolSettings` and exposes provenance solely via logs. (Review
  correction: an earlier draft claimed `/system` shows the calibrated value with
  confidence/baseline; that was inaccurate.)

## Rationale and Alternatives

- **Settings-gated (chosen)** — consistent with the prompt-optimization precedent
  ("controlled entirely via settings, not feature flags").
- **Mutable `ToolSettings` (rejected)** — breaks immutability and the
  "SessionConfig is the single mutation point" mandate.
- **`calibration_status` provenance field on `ToolSettings` (rejected)** — couples
  a config type to runtime observability; read in ~30 sites. Provenance is logged
  instead.
- **Feature-flag gating (rejected)** — inconsistent with the settings-driven
  precedent for runtime-evolution behavior.
- **Per-call dynamic admission control (rejected, out of scope)** — this FEP is
  *session* budget calibration only.

## Backwards Compatibility

100%. Default-off + identity-return means no existing session changes behavior.
The two new settings are optional with safe defaults. No schema migrations
(global DB schema unchanged). No consumption-site edits.

## Security and Privacy

No new data is collected or persisted. The reader queries *existing* RL tables
already governed by the global-database access pattern. Calibration is a pure
read->compute->immutable-overlay; it writes nothing.

## Implementation Status

**Completed and committed (34 tests GREEN across 3 modules):**

| Phase | Commit | Module | Tests |
|-------|--------|--------|-------|
| 1 - Pure policy | `0f6f8db45` | `budget_calibration.py` | 20 |
| 2 - Persistence adapter | `b33ee3c87` | `budget_signal_reader.py` | 8 |
| 3 - Application seam | `b47861ef4` | `budget_calibration_seam.py` + 2 settings | 6 |

Docs parity: `docs/reference/configuration-options.md` documents both settings
(commit `6cca24360`). Decision record: ADR-017.

**Deferred (explicitly blocked on this FEP's review window):**

- The single `AgentFactory.create()` wiring call above.
- An integration test asserting the orchestrator receives the overlay.
- This is the only *runtime behavior change* in the proposal; everything else is
  inert / additive.

## Open Questions

1. ~~Default `min_confidence`~~ — **DECIDED (review #1): 0.7.** `_sample_confidence()`
   returns ~0.63 at the configured minimum sample size and ~0.95 at 3x; combined
   confidence is `0.6*decision_conf + 0.4*tool_conf`. A 0.5 gate would apply as
   soon as both sources barely clear reliability gates; 0.7 requires modestly more
   evidence. Default changed in `tool_settings.py` and docs.
2. **Precedence (review #2): explicit overrides win absolutely.** Resolved by
   accepting an `explicit_override` flag on the seam (short-circuit to
   identity-return when `True`). The prerequisite pre-existing bug is now
   **fixed**: `SessionConfig.tool_budget` is applied to
   `settings.tools.tool_call_budget` inside `apply_to_settings()` (TDD coverage
   in `TestSessionConfigApplyToolBudget`, 3 tests). The real CLI override path is
   now exercisable. The remaining deferred work is the seam's `explicit_override`
   flag wiring + the precedence integration test.

## References

- ADR-017: RL-Driven Tool-Budget Calibration
- ADR-004: Tool System Architecture
- CLAUDE.md - "Prompt Optimization Settings" (settings-vs-flag precedent);
  "When framework changes need more than code" (FEP triggers)
- `victor/framework/rl/budget_calibration.py`,
  `victor/framework/rl/budget_signal_reader.py`,
  `victor/framework/rl/budget_calibration_seam.py`
