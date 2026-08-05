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
2. **Weight independence is what graduates *on the calibration corpus***:
   llama3.3:70b passes every corpus gate (scripted 1.000 at n=96, real 0.878,
   refactor 1.000) against both verifier and annotator gold. **But this did NOT
   transfer** — see point 4.
3. **Grounded training beats nothing yet**: the lexical baseline collapses on
   real trajectories (−0.266 despite 0.930 scripted-dev) — the semantic
   encoder arm was the open question, now answered in point 4.
4. **The calibration corpus was not the shipping distribution (2026-08-05,
   DECISIVE)**: on in-container-verified **SWE-bench-lite** (real GitHub issues,
   17% solve) the corpus gate-passers *over-credit* (gemma4:31b α=−0.52,
   llama3.3:70b α=0.26) and a ModernBERT classifier trained on those real
   trajectories *under-credits* (α=−0.05, majority-class collapse). All three
   fail; only the **in-container verifier** discriminates. **Do not graduate an
   LLM/classifier completion judge as a default** — keep `completion_strategy=
   "enhanced"` + the verifier oracle. Revisit only with a higher-solve-rate
   agent (positives are scarce, not the judge's capacity).

Committed reports: `benchmarks/judge_calibration/labels/run12/reports-*/`;
SWE-bench re-gate: `labels/swe-bench-lite-30/` and `labels/swe-bench-lite-60/`.
FINDINGS rows: runs 12–14 + the two SWE-bench re-gate sections in
[FINDINGS](../../benchmarks/judge_calibration/FINDINGS.md).
