# Vision: Evaluation-Centric Agent Runtime

**Status:** Draft (north star) · **Created:** 2026-06-21 · **Owner:** Vijaykumar Singh
**Companion artifacts:** [FEP-0008](https://github.com/anvai-labs/victor/blob/develop/feps/fep-0008-evaluation-centric-completion.md) · ADR-009/010/011/012 · [Backlog & spec](evaluation-centric-runtime-backlog.md) (the internal `docs/roadmap.md` carries the same epic for local planning)

> Derived from a survey of the local arXiv corpus (49,273 papers) against Victor's current
> evaluation, agentic-loop, and harness code. This doc is the *why* and *where*; the FEP/ADRs are
> the *what* and *how*.

## Thesis

**An agent is a *model + harness*, and the harness's quality is what we can actually engineer.**
The frontier 2026 literature (Harness-Bench `2605.27922`, HarnessFix `2606.06324`, Self-Harness
`2606.09498`) is converging on a single message that also matches Victor's own prior audits
(`agentic_runtime_roadmap_2026-04-27.md`): **stop adding capabilities; close the evaluation loop and
gate every change on it.** Capability is a property of the *(model, harness-config)* pair, measured
by trace-grounded, regression-gated evaluation — not a property of the base model.

Victor already has the hard parts (a research-rooted PERCEIVE→PLAN→ACT→EVALUATE→DECIDE loop, 34 tool
modules, GEPA/MIPROv2 prompt evolution, an offline `AgenticExecutionTrace`, a per-step
`credit_assignment.py` taxonomy). The gap is not *more loop* — it is that **the loop's judgment of
its own work is heuristic, uncalibrated, unvalidated, and not grounded in verifiable effects.**

## Why now

Today's `AgenticLoop._evaluate` decides completion via an algorithmic `EnhancedCompletionEvaluator`
that **under-scores finished answers** (~0.3, "insufficient progress" — observed live), forcing the
loop to burn low-confidence retries and restate the answer. The FEP-0007 cutover work patched this
twice (a HIGH-confidence marker signal and a `_is_terminal_answer` override) — but those are
heuristics layered on a heuristic. The literature offers a principled replacement and, crucially, a
way to *measure whether any of it actually helps*.

Three loops are open or only half-wired:

1. **EVALUATE/completion is a heuristic scalar.** → make it a *calibrated, multi-dimensional,
   effect-grounded, judge-validated* decision.
2. **Victor's parity/characterization batteries are an informal gate.** → make them a *formal
   regression-aware harness-edit acceptance oracle*, reported at *(model, harness-config)* granularity.
3. **Per-step credit is computed but unused** (`credit_assignment.py` exists; the Q-learners consume
   only outcome-level reward). → close it with *segment-level* process reward.

## North star (12-month horizon)

Victor evaluates its own behavior the way a disciplined engineering org evaluates a service:

- **Every completion decision is explainable and calibrated.** The loop stops because named quality
  dimensions cleared calibrated thresholds *and* a verifiable workspace effect exists — not because a
  scalar crossed a hand-tuned bar. (AdaRubric `2603.21362`, effect-grounding from HarnessFix.)
- **Failures are caught at the turn they occur, not post-hoc.** An online, prefix-only auditor flags
  the decisive error and opens an intervention/replan window. (AgentForesight `2605.08715`.)
- **The judge is trusted only after it is measured.** LLM-as-judge quality is reported as
  Krippendorff α / Cohen κ against a human-labeled trajectory set; substring/keyword checks (κ≈0.05,
  chance) are retired. (AgentProp-Bench `2604.16706`.)
- **No harness or prompt edit ships without passing a regression-gated acceptance oracle**, and eval
  results carry confidence intervals at *(model, harness-config)* granularity. (Harness-Bench,
  HarnessFix, Self-Harness.)
- **The runtime learns from its own traces** — segment-level process reward feeds the learners, and
  the harness proposes its own minimal, regression-gated improvements. (HISR `2603.18683`, A²TGPO
  `2605.06200`, Self-Harness.)

## Pillars

| Pillar | From heuristic… | …to evaluated | Key papers |
|---|---|---|---|
| **P1 Completion** | scalar `EnhancedCompletionEvaluator` | task-adaptive rubric dimensions + confidence weights + DimensionAwareFilter | AdaRubric `2603.21362` |
| **P2 Effect-grounding** | textual "looks done" | requires a verifiable artifact/state delta | HarnessFix `2606.06324`, Harness-Bench `2605.27922` |
| **P3 Online auditing** | post-hoc spin/repetition | prefix-only continue/alarm per turn | AgentForesight `2605.08715` |
| **P4 Judge validation** | unmeasured heuristics | κ/α vs human labels; order-swap ensembles | AgentProp-Bench `2604.16706` |
| **P5 Harness-as-gate** | informal batteries | regression-gated acceptance oracle, *(model,harness)* granular | Harness-Bench / HarnessFix / Self-Harness |
| **P6 Closed learning loop** | outcome-only RL | segment-level process reward → learners | HISR `2603.18683`, A²TGPO `2605.06200` |

Supporting (plumbing) directions that the harness then validates: causal-frontier tool selection
(CMTF `2606.06284`), prune/resample/suspend recovery (PruneTIR `2605.09931`), adaptive plan depth
(AdaPlan-H `2604.23194`), belief-entropy compaction guard (MMPO `2605.30159`), dual-rubric context
pruning (LaMR `2605.15315`), isotonic-calibrated routing (UCCI `2605.18796`), cost-aware GEPA
(MO-CAPO `2605.18869`).

## Principles

1. **Measure before adding.** New behavior lands behind the acceptance oracle or not at all.
2. **One canonical loop.** Streaming is an I/O mode of the PPAED loop (FEP-0007), externally
   validated by engine-owns-routing results (GraphBit `2605.13848`). No parallel abstractions.
3. **Calibrated, not heuristic.** Confidence/uncertainty is isotonic-calibrated and reported (ECE),
   not asserted.
4. **Effect over assertion.** "Done" means a verifiable state delta, not a confident sentence.
5. **Reuse before building.** The prior audits cancelled 4 duplicate modules; honor that discipline.

## Non-goals

- LLM-serving infrastructure (KV-cache donation, speculative decoding) — Victor is a framework; it
  only controls prefix stability. Reference only (`2604.05012`).
- Replacing the StateGraph engine or introducing a new orchestration abstraction.
- Training new reward models from scratch where a cached prompt-based judge suffices.

## How this integrates with the existing roadmap

This vision **integrates** (does not replace) the prior arxive roadmap. The standing P0s map in:
cost-aware topology routing and the generative-optimization benchmark harness become *consumers* of
the acceptance oracle (P5); proactive/experiment memory is fed by the closed learning loop (P6);
calibrated uncertainty (truth-aligned `2604.00445`) is the substrate for P1/P4. See the roadmap's
**Evaluation-Centric Runtime** epic for the unified, sequenced plan.

## References

- arXiv: `2603.21362`, `2605.08715`, `2604.16706`, `2606.06324`, `2605.27922`, `2606.09498`,
  `2606.06284`, `2605.09931`, `2605.13848`, `2605.15315`, `2604.23194`, `2605.30159`, `2605.18796`,
  `2605.18869`, `2604.08178`, `2603.18683`, `2605.06200`, `2605.26298`, `2604.00445`.
- Victor: `victor/framework/agentic_loop.py`, `victor/framework/enhanced_completion_evaluation.py`,
  `victor/framework/fulfillment.py`, `victor/agent/turn_policy.py`, `victor/framework/rl/`,
  `victor/agent/credit_assignment.py`, `victor/evaluation/`, `victor/tools/verification/`,
  `tests/integration/streaming/` (parity + characterization batteries).
- Prior audits: `../arxive/agentic_runtime_roadmap_2026-04-27.md`,
  `../arxive/agentic_ai_optimization_audit_2026-04-25_refresh.md`.
