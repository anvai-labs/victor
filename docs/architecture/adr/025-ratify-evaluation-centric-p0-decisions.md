# ADR-025: Ratify the Evaluation-Centric P0 Decisions

## Metadata

- **Status**: Proposed
- **Date**: 2026-07-29
- **Decision Makers**: Vijaykumar Singh
- **Related ADRs**: 009 (rubric completion), 010 (effect-grounded completion), 011 (LLM-judge
  reliability gating), 012 (regression-gated harness acceptance)
- **Work tracked by**: [EVR-4, EVR-5, EVR-2, EVR-3](../evaluation-centric-runtime-backlog.md)
  (existing — this ADR does **not** create new work items)
- **Governed by**: [FEP-0008](../../../feps/fep-0008-evaluation-centric-completion.md)
- **Benchmark**: [competitive-benchmark-2026-07.md](../competitive-benchmark-2026-07.md) §5

## Context

The evaluation-centric runtime is Victor's clearest differentiator: no mainstream agent framework
ships a **graded acceptance oracle** or a calibrated, effect-grounded completion decision (benchmark
§5). The *decisions* already exist as ADR-009/010/011/012 and the vision doc; the *work* is sequenced
in the EVR backlog (EVR-1..17). But two of those ADRs are still **Proposed** and their gates unrun,
so the differentiator is on paper:

- **ADR-010 (effect-grounded completion)** — Proposed, not implemented (EVR-4, P0).
- **ADR-012 (regression-gated harness acceptance)** — Proposed, partial: parity/characterization
  batteries exist but the formal acceptance *oracle* is EVR-5 (P0).
- **ADR-011 (judge reliability)** — Accepted and shipped (`victor/evaluation/judge_calibration.py`),
  but the κ/α gate has **not** been run against a human-labeled set.
- **ADR-009 (rubric completion)** — Accepted, shipped opt-in; default stays `enhanced` pending the
  parity gate.

The risk here is **execution, not competition**: three loops remain open (vision doc — completion is
heuristic, the acceptance battery is informal, per-step credit is unused).

## Decision

Ratify the P0 sequence as a dated, gated commitment — turning the standing "Proposed" ADRs into a
decision with explicit acceptance criteria rather than re-proposing anything:

1. **ADR-010 advances Proposed→Accepted when EVR-4 lands** an effect-grounded completion gate that
   requires a *verifiable workspace state delta* (via `tools/verification/`) before EVALUATE can
   return "done".
2. **ADR-012 advances Proposed→Accepted when EVR-5 lands** the regression-gated acceptance oracle
   reporting at *(model, harness-config)* granularity with confidence intervals; thereafter **no
   harness/prompt edit ships without passing it** (the vision's Principle 1, "measure before adding").
3. **ADR-011's κ/α gate is actually run** (EVR-2) against a human-labeled trajectory set before the
   judge is trusted in any default-on decision; substring/keyword checks (κ≈chance) are retired.
4. **ADR-009's default flip** (`completion_strategy=rubric` becomes default) happens **only after**
   EVR-3 match-or-beats `EnhancedCompletionEvaluator` on the parity + characterization batteries —
   per the sequencing already in the EVR backlog.

Sequencing is the backlog's: `EVR-1 → EVR-2 → (EVR-4 ∥ EVR-3) → EVR-5 → …`. This ADR is the
*decision record* that these gates are the acceptance bar and that the differentiator is a shipping
commitment, not a research aspiration.

## Rationale

- **First principles.** "Done" must mean a *verifiable effect*, and "better" must mean *passed a
  regression-gated oracle* — otherwise the agent's judgment of its own work is a confident sentence.
  This is the vision's Effect-over-assertion and Measure-before-adding principles made into a gate.
- **No new work, no duplication.** Every item maps to an existing EVR entry; this ADR adds a
  *decision* and dates, not a backlog. The "Already delivered — do not re-propose" list in the EVR
  backlog is respected.
- **Differentiator discipline.** Competitors don't have this; the way Victor loses the lead is by
  leaving it half-wired. A dated ratification with hard gates is the counter.

## Consequences

- **Positive**: the eval-loop moat becomes a shipping commitment with objective acceptance criteria;
  ADR-010/012 get a clear path off "Proposed"; default-on decisions become gated, not heuristic.
- **Negative**: the gates are real cost (human-labeled sets, oracle infra, parity runs) and will
  *slow* feature landing by design — that is the intent, not a defect.
- **Neutral**: the PPAED loop and StateGraph engine are unchanged (non-goals in the vision).

## Implementation

No new FEP (FEP-0008 already governs; work is EVR-tracked). This ADR's own lifecycle:

1. On EVR-2 completion → run κ/α, record the number here, retire chance-level checks.
2. On EVR-3 parity pass → flip ADR-009 default; note it here.
3. On EVR-4 → flip ADR-010 to Accepted; update the ADR index Implementation column.
4. On EVR-5 → flip ADR-012 to Accepted; enforce the oracle as a merge gate.
5. When all four are done, flip **this** ADR to Accepted (the P0 sequence is ratified-and-shipped).

## Alternatives Considered

- **Re-propose the eval work as fresh ADRs/TDs.** Rejected: it already exists as ADR-009/010/011/012
  + EVR-1..17; duplicating it violates the backlog's explicit "do not re-propose" discipline.
- **Flip ADR-010/012 to Accepted now.** Rejected: their gates (EVR-4/EVR-5) haven't run; an ADR that
  shipped weeks ago must not read "Proposed", but equally a decision must not read "Accepted" before
  its acceptance criteria are met.
- **Make rubric completion the default immediately.** Rejected: must match-or-beat on the parity
  battery first (EVR-3) — the whole point of measure-before-adding.

## References

- [vision-evaluation-centric-runtime.md](../vision-evaluation-centric-runtime.md),
  [evaluation-centric-runtime-backlog.md](../evaluation-centric-runtime-backlog.md)
- [ADR-009](009-rubric-based-completion-evaluation.md), [ADR-010](010-effect-grounded-completion.md),
  [ADR-011](011-llm-judge-reliability-gating.md), [ADR-012](012-regression-gated-harness-acceptance.md)
- `victor/framework/agentic_loop.py`, `victor/evaluation/`, `victor/tools/verification/`

## Revision History

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2026-07-29 | 1.0 | Initial ADR — ratifies the evaluation-centric P0 gates (EVR-2/3/4/5) | Vijaykumar Singh |
