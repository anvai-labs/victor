# Evaluation-Centric Runtime — Backlog & Spec

**Status:** Draft · **Created:** 2026-06-21 · **Companion:**
[Vision](vision-evaluation-centric-runtime.md) ·
[FEP-0008](../../feps/fep-0008-evaluation-centric-completion.md) · ADR-009/010/011/012

Prioritized backlog for the Evaluation-Centric Runtime direction, derived from an arXiv corpus
survey grounded against Victor's eval/loop code. **Thesis:** *Agent = Model + Harness* — close the
evaluation loop and gate every change on it. Tags: `feature` / `techdebt` / `vision`. Effort:
S/M/L. This **integrates** the prior `../arxive` roadmap (see "Integration" below); the internal
`docs/roadmap.md` carries the same epic for local planning.

| ID | Item | Tag | Pri | Effort | Depends | Papers |
|----|------|-----|-----|--------|---------|--------|
| EVR-1 | Trajectory-eval harness in `victor/evaluation/` (planning / tool-grounding / recovery / refusal, with CI confidence bounds) | feature | **P0** | M | — | TRBench `2604.08178`, `2605.10448` |
| EVR-2 | LLM-judge reliability gate (Krippendorff α / κ vs human-labeled set; order-swap ensemble) — ADR-011 | feature | **P0** | M | EVR-1 | AgentProp-Bench `2604.16706` |
| EVR-3 | Rubric-based completion evaluator + DimensionAwareFilter — ADR-009, FEP-0008 Phase A — **Shipped opt-in; default NO-GO after SWE-bench-lite reliability failure; Prong-B A/B runner shipped for future candidates** | feature | **P0** | L | EVR-2 | AdaRubric `2603.21362` |
| EVR-4 | Effect-grounded completion gate (verifiable state delta) — ADR-010 — **Shipped opt-in** (`victor/framework/effect_gate.py`) | feature | **P0** | M | `tools/verification/` | HarnessFix `2606.06324`, Harness-Bench `2605.27922` |
| EVR-5 | Regression-gated harness acceptance oracle + HTIR/ETCLOVG traces — ADR-012 — **Shipped** (`victor/evaluation/{acceptance_oracle,htir}.py`, promotion-gated) | techdebt | **P0** | M | FEP-0007 batteries | HarnessFix, Self-Harness `2606.09498` |
| EVR-6 | Online per-turn auditor (prefix-only continue/alarm) — FEP-0008 Phase C — **Shipped opt-in; offline HTIR-oracle gate + digest-pinned producer shipped, independently labelled real-distribution pack pending; default OFF** | feature | P1 | M | edge model | AgentForesight `2605.08715` |
| EVR-7 | Close credit→learner loop with segment-level process reward | techdebt | P1 | L | `victor/agent/credit_assignment.py` | HISR `2603.18683`, A²TGPO `2605.06200` |
| EVR-8 | Causal-frontier tool filtering (`requires`/`effects`/`risk` on `BaseTool`) | feature | P1 | M | tool registry | CMTF `2606.06284` |
| EVR-9 | PruneTIR recovery ops (prune-resolved / resample-stuck / suspend-after-repeat) | feature | P1 | M | `RecoveryService` | PruneTIR `2605.09931` |
| EVR-10 | Adaptive plan-depth in PLAN node | feature | P1 | M | `TaskAnalyzer` | AdaPlan-H `2604.23194` |
| EVR-11 | Isotonic-calibrated routing/confidence (ECE-reported) | feature | P1 | M | smart routing | UCCI `2605.18796` |
| EVR-12 | Cost-aware GEPA: add (quality, token-cost) to the Pareto objective | feature | P2 | M | `victor/framework/rl/` | MO-CAPO `2605.18869` |
| EVR-13 | Dual-rubric context pruning (semantic + dependency, graph-derived labels) | feature | P2 | M | graph index | LaMR `2605.15315` |
| EVR-14 | Belief-entropy compaction guard (anchor-question probe) | feature | P2 | S | edge model | MMPO `2605.30159` |
| EVR-15 | Runtime-supervisor sandbox tier via ASK policy (static/runtime split) | feature | P2 | M | sandbox + policy engine | Sandlock `2605.26298` |
| EVR-16 | Speculative tool pre-execution on slack resources (sandbox-gated) | vision | P3 | L | sandbox/policy | PASTE `2603.18897` |
| EVR-17 | Self-improving harness loop (mine → minimal edit → regression-gate) | vision | P3 | L | EVR-5 | Self-Harness `2606.09498` |

## Sequencing (measurement-first)

`EVR-1 → EVR-2 → (EVR-4 ∥ EVR-3) → EVR-5 → EVR-6 → EVR-7 → …`

EVR-3 must **match-or-beat** `EnhancedCompletionEvaluator` on the shipping-distribution reliability,
parity, characterization, and verifier-backed task-success batteries before becoming the default
`completion_strategy`. The 2026-08-05 SWE-bench-lite reliability re-gate failed, so the current
decision is a documented no-go: keep `enhanced` default and revisit only with a new candidate.

## Integration with the prior roadmap

From `../arxive/agentic_runtime_roadmap_2026-04-27.md` (standing P0s), reconciled — not replaced:

- **Cost-aware topology routing** and the **generative-optimization benchmark harness** become
  *consumers* of the EVR-5 acceptance oracle (they are graded by it).
- **Experiment / proactive memory** is fed by EVR-7 (segment-level process reward → learners).
- **Calibrated uncertainty** (truth-aligned `2604.00445`) is the substrate for EVR-2 / EVR-11.

## Already delivered — do not re-propose

Perception + calibrated-confidence fusion, the PPAED loop (FEP-0007 unified), GEPA / MIPROv2 / CoT
distillation, FulfillmentDetector, semantic response cache, paradigm routing, tool-loop/spin
detection, offline `AgenticExecutionTrace`.

**Meta-deliberation narration guard.** `_is_intent_only_response` (in both
`victor/framework/agentic_loop.py` and `victor/framework/enhanced_completion_evaluation.py`)
now performs a full-response density check in addition to the legacy first-line prefix
check. When a response carries no payload (no fenced code block, no markdown table) and
contains 3+ distinct imminent-action markers ("Executing now", "Going now", "Calling now",
"Making the call", "no more deliberation"), it is classified as intent-only narration rather
than a substantial answer. This prevents the failure mode where the model narrates imminent
action without ever invoking a tool, which previously exited the loop before any tool ran.
Real answers carrying a code block or result table are never flagged.
