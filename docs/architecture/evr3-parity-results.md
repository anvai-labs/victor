# EVR-3 Parity Results: Rubric vs Enhanced Completion Evaluation

Status: measured 2026-08-02. ADR-009's graduation condition —
`completion_strategy=rubric` must **match-or-beat** the default
`EnhancedCompletionEvaluator` before it can become the default — is answered
decisively for the judge-quality prong (prong A). The end-to-end task-success
prong (prong B) remains.

## How it was measured (prong A — judge quality)

The `EnhancedCompletionEvaluator` was wrapped as a calibration judge
(`victor/evaluation/calibration_enhanced_judge.py`, the PR-#817 adapter — the
harness whose absence the roadmap previously noted) so it could be scored on
the SAME blinded (prompt, transcript, workspace) views as every LLM judge.
Both were then measured on the run-12 real-agent pack (96 trajectories,
qwen3-coder-tools:30b) via `--replay-pack`, against verifier gold (validated
κ=1.0 vs independent annotation) and the AI-annotator overlay. Reproducible
from the committed labels (`benchmarks/judge_calibration/labels/run12/`).

## Result: rubric (calibrated judge) beats enhanced decisively

| Evaluator | Real-pack α | vs enhanced |
|---|---|---|
| `enhanced` (production default) | **−0.837** | — |
| rubric-llm, gemma4:31b | 0.694 | Δα = **+1.53** |
| rubric-llm, **llama3.3:70b** | **0.878** | Δα = **+1.72** |

On identical trajectories, the rubric path with a calibrated judge
(llama3.3:70b) agrees with truth at α=0.878 while the default `enhanced`
evaluator is **anti-correlated** at −0.837. The gap is not marginal — enhanced
is worse than a coin flip because it reads the agent's live narrative (the
trailing / self-doubting "I was unable to…" final messages that accompany
genuinely-complete work), whereas the rubric judge scores the workspace
evidence. See `docs/architecture/judge-independence-experiments.md` for the
full ladder and [FINDINGS](../../benchmarks/judge_calibration/FINDINGS.md)
runs 11–13.

## What this unblocks and what it does not

- **Unblocks**: the ADR-009 match-or-beat condition is met — rubric does not
  merely match enhanced, it beats it by >1.7 α with a graduated judge. Combined
  with the ADR-011 judge-identity pin (`DEFAULT_CALIBRATED_JUDGE_MODELS` =
  llama3.3:70b), the rubric default-flip (program PR-8) has its evidence: with
  a calibrated judge present, `completion_strategy=rubric` is strictly better;
  without one, the pin falls back to `enhanced` (unchanged behavior).
- **Does not settle (prong B)**: this measures completion-verdict *agreement*,
  not end-to-end *task success* under each strategy in a live loop. Before the
  default flips, prong B should confirm no task-success regression and no
  completion-latency blowout in a flag-on/flag-off A/B (the EVR-4-style battery),
  and the streaming byte-stability batteries must stay green flag-off. Prong A
  establishes the judge is right; prong B establishes the loop is not worse.

## Honest caveats

- Single corpus (6 templates), single agent distribution. The SWE-bench stratum
  is the named prerequisite for a beyond-corpus parity claim.
- The rubric advantage is *judge-specific*: it holds for calibrated judges
  (llama3.3:70b; gemma4:31b clears enhanced but not the 0.7 gate) and collapses
  for uncalibrated models — which is exactly what the ADR-011 pin enforces.
