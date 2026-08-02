# Trained Judge Experiments (E2) — Findings

Small independent completion judges trained on programmatic verifier gold
(κ=1.0-validated, FINDINGS run 12). Dataset: `generate_dataset.py` (8,400
scripted examples, 7,200 train / 1,200 dev, split by variant index). The gate
is the **held-out run-12 real-agent pack** (96 trajectories) via
`run_offline_calibration.py --replay-pack --{classifier,encoder}-judge` +
`--human-labels`, thresholds identical to every LLM judge (α ≥ 0.7 overall and
per n≥16 family). See `docs/architecture/judge-independence-experiments.md`.

| Arm | Model | Dev α (scripted) | Real-pack α | Verdict |
|---|---|---|---|---|
| A | hashing-trick logistic (`victor/ml/features`, sklearn) | 0.930 | **−0.266** | FAIL |
| B | ModernBERT-base (149M, HF) | **1.000** | **−0.266** | FAIL |

## The decisive result: identical failure ⇒ distribution shift, not capacity

Both arms score **exactly −0.266** on the real pack despite near-perfect
scripted dev scores. The encoder's confusion on the 96 real trajectories:
**TP=0, FN=88, TN=8, FP=0** — it labels *every* trajectory "not complete." It
scores the 8 genuine failures correctly only by rejecting everything, and
calls all 88 real completions incomplete.

Diagnosis: the scripted `solve` executor produces clean, terse completions;
real agent completions carry 20-step tool logs, extra scratch files, and
trailing / self-doubting final messages ("I was unable to…" while the
workspace shows the work done — the same surface that sank gemma4:31b's
refactor family and `enhanced`). A model trained to recognize *scripted*
completion learns features that are anti-correlated with *real* completion.
That a 149M semantic transformer and a linear bag-of-features model fail to
the identical number proves the bottleneck is the **training distribution**,
not model class or capacity.

## Consequence for the pre-registered escalation

The plan pre-registered: encoder fails the real-pack gate after 2 iterations →
escalate to Qwen2.5-1.5B QLoRA **on the same data**. That criterion is met, but
the identical dual-arm failure re-diagnoses the problem: a bigger model on the
same scripted distribution is predicted to fail the same way. The evidence
redirects the next iteration from *more capacity* to *real-distribution
training data* — either the SWE-bench stratum (named in the plan as the
prerequisite for any beyond-corpus claim) or a dedicated real-agent training
run held strictly separate from the eval pack. This is a plan-level decision,
recorded here rather than spent on a predictable-failure training run.

## What stands

For in-loop / per-turn judging today, the graduated judge is **llama3.3:70b**
(FINDINGS runs 10/13). The small-judge track remains the right *economics* bet
for EVR-6 per-turn auditing, but it must be trained on real-distribution data
before it can clear the same gate — the reusable pieces (dataset generator,
`make_classifier_judge` wrapper, `--classifier/encoder-judge` gate hooks,
pack-replay comparability) are all in place for that next iteration.
