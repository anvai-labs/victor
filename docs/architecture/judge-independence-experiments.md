# Judge Independence: Options and Measured Evidence

Status: living experiment log (2026-08-02 —). Protocol:
[EVR-2 human-validation protocol](evr2-human-validation-protocol.md) (Amendment 1:
AI-annotator gold). All rows measured on the SAME 96 real agent trajectories
(run 12 pack, qwen3-coder-tools:30b agent) against verifier gold (validated
κ=1.0 vs independent annotation) — reproducible via
`--replay-pack` + `--human-labels` on the committed labels
(`benchmarks/judge_calibration/labels/run12/`).

## The option ladder

| Tier | Independence from the agent | α (real pack) | Verdict |
|---|---|---|---|
| `enhanced` (in-loop self-assessment) | none — reads the live trace | **−0.837** | anti-correlated; run 12 |
| Fresh-context same model (self-judge, current rubric wiring) | context only | **0.469** (qa −0.03, refactor 0.10) | FAILS — run 14 |
| Cross-model gemma4:31b | context + weights | **0.694** (refactor 0.30) | FAILS at n=96 — run 12; demoted |
| Cross-model llama3.3:70b | context + weights | **0.878** (refactor 1.00) | **PASSES — run 13; the pinned judge** |
| Linear classifier on verifier gold (E2 arm A) | weights + grounded training | **−0.266** (dev-scripted 0.930) | FAILS — lexical features don't transfer |
| ModernBERT encoder (E2 arm B) | weights + grounded + semantic | pending | training |
| ~1.5B QLoRA (E3) | as above + generative capacity | not started | pre-registered escalation only |

## What the ladder establishes

1. **Context isolation is necessary but insufficient**: fresh context recovers
   1.3 α over in-loop self-assessment, yet weight-level correlation still costs
   ~0.41 α vs an independent judge on identical trajectories. Session-model
   self-judging cannot enter the calibrated set.
2. **Weight independence is what graduates**: llama3.3:70b passes every gate
   (scripted 1.000 at n=96, real 0.878, refactor 1.000 — the family that
   defeats claim-readers) against both verifier and annotator gold.
3. **Grounded training beats nothing yet**: the lexical baseline collapses on
   real trajectories (−0.266 despite 0.930 scripted-dev) — the semantic
   encoder arm is the open question.

Committed reports: `benchmarks/judge_calibration/labels/run12/reports-*/`.
FINDINGS rows: runs 12–14 in
[FINDINGS](../../benchmarks/judge_calibration/FINDINGS.md).
