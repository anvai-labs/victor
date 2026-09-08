# ADR-011: LLM-Judge Reliability Gating

## Metadata

- **Status**: Accepted (2026-07-02 — machinery shipped in `victor/evaluation/judge_calibration.py` + `trajectory_eval.py` (#202); the κ/α validation against human labels has not yet been run, so ADR-009/010 stay opt-in; was Proposed)
- **Date**: 2026-06-21
- **Decision Makers**: Vijaykumar Singh
- **Related ADRs**: 009 (rubric completion), 010 (effect-grounded), 012 (regression-gated harness)
- **Related**: [FEP-0008](https://github.com/anvai-labs/victor/blob/develop/feps/fep-0008-evaluation-centric-completion.md)

## Context

ADR-009/010 introduce LLM-as-judge into the completion decision. AgentProp-Bench (`2604.16706`)
measured judge reliability empirically: substring/keyword matching agrees with humans at **κ=0.049
(chance)**; a 3-LLM order-swapped ensemble reaches κ=0.432 (moderate). It also shows **rejection**
(catch bad input) and **recovery** (fix after accepting) are *statistically independent* (ρ=0.126).
"Time to REFLECT" (`2605.19196`) and benchmark-confidence work (`2605.10448`) reinforce: do not
trust a judge you have not measured. Victor currently has **no judge-reliability measurement
anywhere**, and any substring/keyword completion heuristic is, per the evidence, chance-level.

## Decision

No LLM-as-judge (rubric scoring, trajectory grading) is trusted in production until it passes a
**reliability gate**: maintain a small human-labeled trajectory set, compute **Krippendorff α /
Cohen κ** against it, and enable the judge only at **α ≥ threshold** (default 0.7, configurable);
otherwise fall back to algorithmic scoring. Judges run as a small **order-swap ensemble** (≥2
reorderings) to defuse position/verbosity bias. **Rejection** and **recovery** are tracked as
separate metrics, never folded into one "robustness" score.

## Rationale

- Prevents shipping a confident-but-chance-level judge — the failure mode the literature documents.
- The α-gate is the precondition that makes ADR-009/010 *safe to default-on*: rubric/effect judging
  is enabled only once measured trustworthy on this corpus.
- Order-swap ensembling is a cheap, training-free bias mitigation with documented κ gains.
- **Pros**: honest evals; safe judge adoption; reusable calibration asset. **Cons**: requires
  building/maintaining a labeled set (small, amortized; seed from existing trajectories + spot labels).

## Consequences

- **Positive**: judge trust is earned and reported; substring/keyword completion heuristics retired;
  separate rejection/recovery metrics give actionable per-turn signal.
- **Negative**: labeling effort and an ensemble cost (≥2 calls) when the judge is active.
- **Neutral**: does not change the loop control flow; it gates *which evaluator* is used.

## Implementation

1. `victor/evaluation/judge_calibration/` with a seed human-labeled trajectory set + α/κ computation.
2. A reliability gate consulted by `RubricCompletionEvaluator` / trajectory harness; α < threshold →
   algorithmic fallback.
3. Order-swap ensemble wrapper; report α/κ and rejection/recovery in eval output and CI.
