# ADR-013: Unified, Intent-Based Temperature Policy with Spin Ratchet

## Metadata

- **Status**: Accepted (2026-07-02 — implemented in `victor/framework/temperature/` with the 0.7→0.6 default flip and the `test_no_temperature_scatter.py` boundary guard; was Proposed)
- **Date**: 2026-06-22
- **Decision Makers**: Vijaykumar Singh
- **Related ADRs**: 009 (rubric completion) — shares the reproducibility concern
- **Related**: [FEP-0007](https://github.com/anvai-labs/victor/blob/develop/feps/fep-0007-unified-agentic-loop.md) (spin/restatement), arXiv 2606.01451, 2603.21301, 2606.13982

## Context

Sampling-temperature logic is scattered across **six** independent paths that drift and duplicate:

1. `ProfileConfig.temperature` base scalar (`config/settings.py:368`, default 0.7).
2. Per-task constants on `TaskTypeHint.temperature_override` (`framework/capabilities/task_hints.py`:
   debug 0.1 … analyze 0.6) — applied **only in the buffered loop** (`agentic_loop.py:858-860`→`2313`→
   seam `turn_execution_runtime.py:1602`).
3. A hardcoded `+0.2` analysis bump that mutates the shared scalar (`agent/services/task_runtime.py:483-489`).
4. Reactive failure escalation (`agent/recovery/temperature.py` `ProgressiveTemperatureAdjuster`:
   `STUCK_LOOP +0.2`, `REPEATED_RESPONSE +0.25`, Q-learning, `MODEL_TEMPERATURE_RANGES`).
5. Streaming **ignores task-hint temperature** entirely — uses `orch.temperature`
   (`agent/services/chat_stream_helpers.py:1337`), so buffered and streaming disagree.
6. Streaming has its **own** hardcoded recovery ramp ladder (`agent/streaming/handler.py:883-935`).

This produces inconsistent behavior between modes, duplicated escalation, and no single place to tune
temperature by intent or per profile. Two needs prompted the change: (a) per-profile, per-task
temperatures (e.g. `glm-5.2 plan=0.5` vs `opus plan=0.55`) with a sane fallback chain; and (b) a
**spin-escape ratchet** — when the loop is stuck in a repetition/plateau, raise temperature modestly to
give the model "escape velocity," reusing the FEP-0007 spin signals.

**Research grounding.** Full-text reading of arXiv `2606.01451` (distributional view of temperature)
shows T≈0.3–0.8 is a coherence-flat region (per-token-identical) and the **degeneration cliff is past
T≈1.0 toward 1.5** (mass on the pre-temperature top-90% set drops 0.982→0.868; `n95` inflates ~1→131
tokens). `2603.21301`/`2604.17433` use T≈0.7–0.8 as the diversity sweet spot. ⇒ a ratchet must use a
**modest step and a hard cap well below 1.0**.

## Decision

Introduce one composition-based **`TemperatureResolver`** in `victor/framework/temperature/` that all
call sites route through. It is built from two narrow Protocols (ISP):

- **`TemperatureSource.resolve(request) → Optional[float]`** — resolves a *base* from static inputs,
  or `None` to defer (Chain of Responsibility). Ordered precedence:
  `ProfilePerTask › SettingsPerTask › TaskHintConstant › ProfileBase › GlobalDefault(0.6)`.
- **`TemperatureModifier.adjust(value, request, context) → (value, reason)`** — idempotent. Applied in
  order `SpinRatchet → RecoveryAdjust → ModelBounds`.

The **`SpinRatchetModifier`** adds `steps × 0.05` (steps advanced once per turn by
`RatchetState.record_turn` on `SpinState.WARNING/STUCK` or plateau; reset on progress), **hard-capped
at 0.9**. The **`RecoveryAdjustModifier`** wraps the existing `ProgressiveTemperatureAdjuster` (via a
framework-side `ReactiveTemperatureAdjuster` Protocol, so framework does not import agent), unifying
the reactive path instead of duplicating it. The **`ModelBoundsModifier`** clamps to
`MODEL_TEMPERATURE_RANGES` (relocated to `framework/temperature/defaults.py`; recovery imports it back).

Per-profile per-task temperatures come from a new optional `ProfileConfig.temperatures` map; a
settings-level `temperature.task_defaults` table provides ops tuning; the existing `TaskTypeHint`
constants remain the SDK-stable floor (read in place — `victor-contracts` unchanged).

## Rationale

- **Consolidation over proliferation**: 6 scattered paths → 1 resolver. `task_runtime.py:483-489` and
  the streaming ramp ladders are deleted/absorbed; the recovery adjuster is *wrapped*, not rebuilt.
- **SOLID**: SRP (one rule per source/modifier), OCP (extend by appending in `factory.py`; resolver
  untouched), LSP (uniform Protocol contracts), ISP (static `TemperatureRequest` vs dynamic
  `TemperatureContext`), DIP (resolver depends on Protocols; `ReactiveTemperatureAdjuster` inverts the
  framework→agent dependency).
- **Framework-first**: the resolver is provider-agnostic and consumed by both the buffered and
  streaming seams, so it lives in `framework/`, closing the buffered/streaming inconsistency at one
  seam.
- **Reproducibility** (ties to the evaluation-centric direction): ground/audit/judge intents resolve
  low; the ratchet resets on progress so it does not inflate eval variance.

## Consequences

- **Behavior change**: the global default moves 0.7→0.6 for profiles without an explicit temperature.
  Gated by `temperature.global_default`; the buffered cutover kept 0.7 first to isolate. **Flipped to
  0.6 on 2026-06-22** after a multi-turn A/B (`completion_ab.py`/`temperature_ab.py`, glm-5.2, honest
  turn metric via #235): `enhanced@0.60` was the best-combo (5.2 turns, all-complete) and 0.60
  match-or-beat 0.70 on the temperature sweep (5.30 vs 6.30 turns, equal variance). Applied to
  `ProfileConfig.temperature`, the built-in BASIC/ADVANCED/EXPERT profile presets, and the
  `ProviderSettings.default_temperature` defaults. Quality was not auto-scored (efficiency gate only).
- The `temperature_override` plumbing in the loop becomes redundant and is removed once both seams
  route through the resolver.
- An AST boundary guard forbids raw `temperature + <float>` arithmetic outside
  `framework/temperature/` and `recovery/temperature.py`, preventing a seventh scatter.
- Q-learning state stays consistent because the same `ProgressiveTemperatureAdjuster` instance is DI-
  shared between the recovery coordinator and the `RecoveryAdjustModifier`.

## Rollout

Five focused PRs: (A) the additive `framework/temperature/` package + this ADR; (B) settings +
`ProfileConfig.temperatures` + DI; (C) buffered cutover + delete the `+0.2` bump; (D) streaming cutover
+ absorb the ramps + buffered↔streaming parity test; (E) ratchet activation + boundary guard.
