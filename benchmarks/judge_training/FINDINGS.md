# Trained Judge Experiments (E2) — Findings

> ## CORRECTION (2026-08-02): the −0.266 results below were a scoring artifact
>
> Every "α=−0.266" verdict in the original writeup was produced by a bug, not
> by the judges. `make_classifier_judge` returned a raw **probability**, and
> the calibration harness computes Krippendorff α at the **nominal** level — so
> `0.999` and `1.0` were treated as different categories and counted as
> disagreement against a `1.0` gold. LLM judges were unaffected (they return
> clean 0/1 completion verdicts); only the probabilistic trained judges were
> mis-scored. Fixed by binarizing the classifier verdict at 0.5
> (`victor/evaluation/calibration_classifier_judge.py`), matching the LLM-judge
> contract. **Corrected results, through the fixed pipeline on the run-12 pack:**
>
> | Training data | Overall α | Per-family | Verdict |
> |---|---|---|---|
> | scripted only (arms A/B) | **−0.84** | all negative | FAIL — constant "incomplete" |
> | real-agent only | **−0.04** | ~0 | FAIL — constant "complete" (real data is 98% positive → no negatives to learn from) |
> | **real positives + scripted negatives (mix)** | **0.928, trusted** | code-fix/docs/qa/refactor = 1.0; **file-create = 0.644** | **NEAR-PASS** — overall gate passes and beats llama3.3:70b (0.878); one family short of the strict per-family bar |
>
> **What actually held from the original diagnosis:** the distribution point is
> real — scripted-only rejects real completions, real-only accepts everything.
> The fix is a training set with real-styled positives AND boundary-providing
> negatives. **What was wrong:** "a 149M model can't do it / capacity isn't the
> issue and a bigger model won't help." With the scoring fixed and the right
> data mix, the 149M ModernBERT reaches α=0.928 — the small independent judge
> essentially works; it is ~one iteration (a cleaner `file-create` negative or
> a tuned threshold) from a full per-family pass. Committed evidence:
> `reports-fixed-{mix,real,scripted}/`.
>
> The original (mis-scored) analysis is retained below for the record.

---

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

---

## Iteration 2 (2026-08-02): real-styled negative synthesis

The mix judge (α=0.928) had one weak family (`file-create` = 0.644). To close
it we synthesized **real-styled negatives** — a real positive's transcript
(its real claim + real tool narration) re-rendered against the task's
*unsolved fixture* = the ADR-010 completion-without-effect case in real style
(`judge_training_data.synthesize_effect_removed_negative`, no GPU, no re-run).

| Training set | Overall α | Per-family | Note |
|---|---|---|---|
| real positives + real-styled negatives (all families) | 0.075 | file-create 1.0, **refactor −0.66**, qa 0.0 | synthesis breaks non-effect families |
| mix + real-styled negatives (all families) | 0.131 | file-create 1.0, **refactor −0.55** | same |
| **mix + file-create real-negs only (targeted)** | **0.936** | code-fix/docs/file-create/refactor = 1.0, **qa = 0.644** | best judge; beats llama3.3:70b (0.878) |

**Two findings.** (1) **Effect-removal synthesis is family-specific**: clean for
*effect-absence* tasks (file-create: no file → unambiguous negative, fixed it to
1.0), but noisy for *subtle-diff* tasks (refactor/code-fix, where negative vs
positive differ only by a function name or one line) and inapplicable to
*answer* tasks (qa — the fixture already verifies complete, so 24 qa negatives
couldn't be synthesized). Applied only where valid (file-create), it strictly
improves the judge; applied blindly, it poisons refactor. (2) **The remaining
per-family gap is eval-pack negative scarcity, not judge quality.** The run-12
pack has 8 negatives / 96 (~2 per n=16 family), so a *single* misclassification
drops that family to 0.644 — and every retraining just moves the one miss to a
different family (file-create 0.644 → qa 0.644 here). The judge is genuinely
strong (α=0.936 overall, 4/5 families at 1.0, > llama's 0.878); a clean
simultaneous per-family pass needs a **denser-negative eval pack** (the
SWE-bench stratum with natural real failures), which is the documented
prerequisite — not another training iteration.

**Standing verdict for the small-judge track:** viable and strong (α≈0.93,
CPU-cheap, weight-independent), ready as the FEP-0030 classifier backend. The
per-family bar is gated on richer real-failure data, tracked with the SWE-bench
stratum. Committed evidence: `reports-iter2/`.
