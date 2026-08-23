# EVR-3 Parity Results: Rubric vs Enhanced Completion Evaluation

Status: calibration-corpus result measured 2026-08-02; default decision
superseded by the 2026-08-05 SWE-bench-lite re-gate. The result below remains
valid for its corpus, but neither calibrated LLM judge cleared reliability on
Victor's shipping distribution. `completion_strategy` therefore remains
`enhanced`; rubric stays opt-in. The Prong-B task-success runner shipped later
as future re-evaluation machinery, not as authorization for a default flip.

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

## What this establishes and what it does not

- **Establishes on this corpus**: rubric+llama3.3:70b beats enhanced by >1.7 α,
  and the ADR-011 identity pin safely falls back to `enhanced` when its premise
  is absent.
- **Does not clear the production gate**: the later in-container-verified
  SWE-bench-lite stratum measured llama3.3:70b at α=0.26 and gemma4:31b at
  α=−0.52. Both fail the ≥0.7 requirement; only the programmatic verifier
  discriminated. See [FINDINGS](../../benchmarks/judge_calibration/FINDINGS.md)
  and [judge independence experiments](judge-independence-experiments.md).
- **Prong B is necessary but not sufficient**: a task-success pass cannot
  override a failed judge-reliability prerequisite.

## Prong-B executable battery

`victor.evaluation.completion_strategy_ab` runs the same verifier-backed tasks
through `enhanced` and `rubric`, preserving paired per-task outcomes plus the
inner loop's completion claim and iteration count. The default gate requires:

- 24 evaluable pairs, with at least four per task family;
- candidate task success to match-or-beat baseline;
- no increase in false-positive completions; and
- mean loop iterations within 10% (or 0.25 iterations) of baseline.

Example with the pinned judge on an independent endpoint:

```bash
python -m victor.evaluation.completion_strategy_ab \
  --model qwen3-coder-tools:30b \
  --base-url http://localhost:11434 \
  --judge llm:llama3.3:70b@http://localhost:11434 \
  --variants 4 --out-dir artifacts/evr3-prong-b
```

The command writes `completion_strategy_ab.json` and returns `pass` or `hold`
for Prong B only. It never edits configuration and never reports `graduate`.
Given the SWE-bench-lite reliability failure, a `pass` today is diagnostic;
the default remains `enhanced`.

## Honest caveats

- Single corpus (6 templates), single agent distribution. The subsequently run
  SWE-bench stratum failed, proving this caveat was material.
- The rubric advantage is *judge- and distribution-specific*: llama3.3:70b
  cleared the calibration corpus but failed SWE-bench-lite. The ADR-011 pin
  still prevents silent use of the known-bad heuristic; it is not evidence for
  a default flip.
