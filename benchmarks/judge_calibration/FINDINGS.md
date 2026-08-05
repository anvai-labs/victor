# Judge-Calibration Findings — EVR-2 / ADR-011 first live measurements

**Date**: 2026-07-02/06 (11 runs) · **Corpus**: `default_calibration_corpus(variants=8)` — 48 tasks,
6 families · **Executor**: scripted (`alternating_scripted_executor(period=5)`, ~38 solved /
~10 completion-without-effect fakes) · **Gate**: Krippendorff α ≥ 0.7 (binary completion
verdicts vs programmatic verifier gold) · **Judges measured**: local Ollama (offline,
runs 1–5, 7) and cloud DeepSeek (run 6); all cloud/paced runs integrity-verified

## Run series

| # | Judge | View / code state | Overall α | Verdict | Diagnosis |
|---|-------|-------------------|-----------|---------|-----------|
| 0a | credulous (claim-trusting baseline) | — | **−0.188** | NOT TRUSTED | Reproduces the AgentProp-Bench result locally: trusting agent claims agrees with ground truth at worse than chance. |
| 0b | evidence (tool-activity baseline) | — | **1.000** | TRUSTED | On scripted trajectories, tool-activity presence is a perfect completion signal (will not hold for real agents that work and still fail). |
| 0c | rubric-heuristic (ADR-009 no-LLM fallback) | names-only view | **0.852** | TRUSTED overall | **At chance (α=0.000) on code-fix** — the family where solved/unsolved workspaces differ only semantically. Overall-α gating would wrongly bless it; per-family gating catches it. |
| 1 | rubric-llm, qwen2.5-coder:14b | names-only view | **0.450** | NOT TRUSTED | 8/10 errors were missed completions, 6 on code-fix: the judge could not assess correctness because the view showed file *names* without *contents*. |
| 2 | rubric-llm, qwen2.5-coder:14b | + workspace file contents | **0.457** | NOT TRUSTED | code-fix fixed (−0.406 → 0.615) — confirming run 1's diagnosis — but docs-link regressed (5 missed completions: judge believed the "right" fix was creating the missing doc, not repointing the link). Failure moved, α unchanged. |
| 3 | rubric-llm, qwen2.5-coder:32b | same view as run 2 | **−0.237** | NOT TRUSTED | Zero false completions, 30 missed: mass refusal. Probing the raw output exposed the scripted executor's claim echoing a mid-word-truncated prompt — the stronger model *correctly* penalized the garbled message the weaker model glossed over. Label poisoning, not judge failure. |
| 4 | rubric-llm, qwen2.5-coder:32b | clean claim message | **−0.412** | NOT TRUSTED | Still mass refusal (34 missed, 0 false). Raw output on a perfect task: three dimensions 1.0@1.0, then `recovery: score=0.0 confidence=0.3` — the un-engaged axis scored just above the DimensionAwareFilter engagement floor (0.25), gating completion. **Framework bug (ADR-009), fixed in `82e20be9`.** |
| 5 | rubric-llm, qwen2.5-coder:32b | + engagement-convention fix | **0.173** | NOT TRUSTED | Large improvement (−0.412 → 0.173): file-create and qa now α=1.000, zero false completions. Remaining 19 misses concentrate on code-fix/docs-link/dead-code. Probing shows the convention fix took (recovery now 0.0@0.0) and the judge *sees* the correct workspace (`tool_grounding: 0.8 — "workspace state shows the correct implementation"`) — but penalizes correctness/completeness because the terse scripted claim ("Done — I completed the requested task.") **doesn't narrate the fix**. Near-identical variants flip between perfect and penalized grades at temperature 0 (variant 0 graded 1.0/1.0/1.0; variant 1 graded 0.5/0.8/0.6), so the residual is part scripted-transcript artifact, part judge instability on narration-free claims. |
| 6 | rubric-llm, **deepseek-chat (cloud)** | run-5 code state; `--judge-delay 5`; first clean cloud run (integrity: calls=48 retries=0 failures=0) | **0.279** | NOT TRUSTED | **New best**, and the first cross-provider data point. Per-family fingerprint nearly identical to run 5: file-create 1.000, qa 1.000, refactor 0.372, code-fix −0.190, docs −0.500 — and again **zero false completions**; all 15 errors are missed completions on solved docs-link (6/7), code-fix (5/7), and dead-code (4/7) tasks. Two unrelated judges (local qwen-32b, cloud DeepSeek) producing the same family-level failure signature confirms the bottleneck is the **narration-free scripted transcripts**, not judge capability — the judges refuse to certify work nobody described, which is defensible behavior against a corpus artifact. Preceding runs on this machine also validated the new guardrails live: a fully rate-limited GLM-5.2 attempt was correctly VOIDed (α would have equaled the heuristic's to 3 decimals), and a DeepSeek attempt exposed the per-call-event-loop bug (PR #394). |
| 7 | rubric-llm, **gemma4:31b (LOCAL, Ollama on Apple Silicon)** | run-5 code state; default 2 s pacing (~84 s/call inference); integrity: calls=48 retries=0 failures=0 | **0.929** | **TRUSTED — first gate pass** | 47/48 verdicts correct. Per-family: code-fix **1.000** (the family that broke every prior judge), docs 1.000, file-create 1.000, qa 1.000, refactor 0.823 — clears the gate overall AND per-family. The single error is the series' **first false completion** (refactor-rename-02: unsolved rename judged complete) — a changed error polarity worth watching at larger n. This result **overturns run 5/6's corpus-ceiling diagnosis**: gemma4 passed on the same narration-free scripted claims that capped qwen-32b and DeepSeek, proving the workspace-state evidence in the view was sufficient all along and the earlier judges' refusals were model disposition, not a corpus artifact. The gate-passing judge is fully local — no API cost, no rate limits, no data egress. |
| 8 | rubric-llm, qwen3.5:27b-q4 (local) | run-7 code state; integrity: calls=48 retries=0 failures=0 — but see diagnosis | **−0.092** | NOT TRUSTED (should be VOID) | **Third integrity blindspot found**: every verdict was 1.0 — 10 false completions, 0 misses — matching rubric-heuristic's α sample-for-sample, the signature of a judge contributing zero signal. Transport succeeded, but no `score=` line ever parsed: every dimension fell back to 0.5@≤0.2 (below the engagement floor), so the DimensionAwareFilter defaulted every task to COMPLETE. Suspected cause: qwen3.5 is a thinking-family model that exhausted the 512-token grading budget on reasoning preamble (~72 s/call supports this). Fixed: `JudgeCallStats.ungradable` now counts all-fallback results and VOIDs the run; `--judge-max-tokens` (default raised to 1024) gives thinking judges headroom. Retest qwen3.5 with `--judge-max-tokens 2048` to distinguish format-mismatch from truncation. |
| 9 | rubric-llm, **llama3.3:70b (local, Ollama on the Windows-host GPU)** | run-8 code state (`--judge-max-tokens 1024`, ungradable guard active); integrity: calls=48 retries=0 failures=0 ungradable=0 | **1.000** | **TRUSTED — perfect score** | 48/48 verdicts correct, zero errors of either polarity, every family at α=1.000. Second gate-passer (of six judge models measured), confirming gate-passing judges are not rare among strong models — and the first ceiling hit: a perfect score at n=48 means this corpus can no longer discriminate among top judges. Ranking so far: llama3.3:70b 1.000 > gemma4:31b 0.929 > deepseek-chat 0.279 > qwen2.5-coder:32b 0.173 ≫ qwen3.5:27b (ungraded). Run also validated the run-8 ungradable guard end-to-end in a live measurement (`ungradable=0` reported). |
| 10 | rubric-llm, **llama3.3:70b (local)** — **variants-16 CONFIRMATION** | run-9 code + workspace-cleanup; n=16/family (96 tasks); integrity: calls=96 retries=0 failures=0 ungradable=0 | **1.000** | **TRUSTED — gating-grade** | 96/96 correct, α=1.000 on every family, zero errors of either polarity. Doubles the sample size of run 9 and holds perfect — this is the gating-grade evidence (n≥16/family) FINDINGS required before per-family α counts. **Graduation-checklist item 1 is satisfied for llama3.3:70b.** Decisively separates the two gate-passers: llama3.3:70b is confirmed perfect at n=96, while gemma4:31b has only n=8 evidence with one false completion — llama3.3:70b is the pinned judge candidate. |
| 11 | rubric-llm, **gemma4:31b** — **REAL AGENT TRAJECTORIES** (checklist item 2) | agent=qwen3-coder-tools:30b via VictorAgentAdapter, two-phase, `--judge-max-tokens 2560`; integrity: calls=48 retries=0 failures=0 ungradable=0 | **0.865** | **TRUSTED — the graduation gate** | The first calibration on REAL agent behavior instead of scripted stand-ins. 46/48 correct; per-family code-fix/docs/file-create/qa all 1.000, refactor 0.754 (all ≥ 0.7); **zero false completions**, 2 missed (both dead-code). Gold was a real 40/8 mix from the agent's own successes and failures (81% solve rate; the 30B agent also hit an `edit`-tool arg-format bug — `ops` sent as str not array — a real capability limit, not a harness fault). The headline contrast: on these SAME real trajectories the scripted-perfect baselines collapsed — `evidence` (tool activity) went 1.000→**−0.092**, `rubric-heuristic` 0.852→−0.092 — because real agents produce tool activity while failing. gemma4:31b held at 0.865. This is the production distribution the gate exists for, and gemma4:31b clears it — **checklist item 2 satisfied.** Two harness fixes made this run possible: real-agent mode + two-phase scheduling (one model swap, ~40 min vs a killed 3 h interleaved run), and the ungradable guard caught a first attempt (`ungradable=9` at 1024 tokens — real transcripts are longer) and VOIDed it before this clean retry. |

## What the series established

1. **The gate works.** Every NOT TRUSTED verdict above was correct — each traced to a real,
   specific defect (in the judge, the view, or the harness itself), not measurement noise.
   And it passes when a judge deserves it (run 7).
2. **Per-family α is mandatory.** Run 0c passes the overall gate while blind on code-fix;
   runs 1–2 show failures migrating between families as the view changes. Gate per family,
   with n ≥ 30 per family (`--variants 16`+) before treating per-family α as evidence.
3. **Stronger judges surface weaker harnesses.** Runs 3–4 both looked like "the 32b model is
   worse." Both times the model was right and the harness was wrong (truncated claim echo;
   engagement-floor mismatch). Diagnose mass-refusal patterns (0 false completions, many
   missed) by probing raw judge output before blaming the model.
4. **Judge choice dominates.** Runs 5–7 ran identical code, views, and corpus; α spanned
   0.173 → 0.279 → 0.929 purely by model. A shared failure pattern across two judges
   (runs 5–6) looked like a corpus ceiling until a third judge broke through it — beware
   concluding "artifact" from N=2 judges.
5. **Fixes found in framework code, not just calibration code** (commit `82e20be9`):
   - Grading prompt now states the numeric not-applicable convention
     (`score=0.0 confidence=0.0`) instead of "use a LOW confidence" — judges' idea of low
     (0.3) sat above the filter's engagement floor (0.25).
   - Unparseable judge rows default to confidence 0.2 (below the floor) so an ungradable
     line cannot gate completion.
   - Scripted-executor claims no longer echo truncated prompts.

## Reproduction

```bash
python benchmarks/judge_calibration/run_offline_calibration.py \
    --variants 8 \
    --llm-judge-provider ollama --llm-judge-model qwen2.5-coder:32b \
    --llm-judge-base-url http://<ollama-host>:11434   # e.g. WSL → Windows gateway
```

Reports (per-family α, gate decision, every sample) land in
`benchmarks/judge_calibration/reports/*.json`.

### Run 12 (2026-08-02) — AI-annotator gold overlay + 5-judge real-trajectory sweep

Fresh 96-trajectory real-agent run (qwen3-coder-tools:30b-64K, two-phase, integrity clean:
calls=96 retries=0 failures=0 ungradable=0; gold 88/8) with the PR #810 labeling-pack export,
labeled per [protocol Amendment 1](../../docs/architecture/evr2-human-validation-protocol.md)
(AI-annotator gold — two independent Claude passes, blind pack view, zero disagreements;
**NOT human gold**). Committed artifacts: `labels/run12/` (labels + all reports + overlay).

| Check | Result |
|---|---|
| annotator↔verifier κ | **1.0000 PASS** (≥ 0.8) — programmatic verifier gold fully validated; runs 1–11 foundation stands |
| annotator↔secondary κ | 1.0000 (QC) |
| rubric-llm gemma4:31b | **α=0.6936 FAIL** (< 0.7); per-family 1.000 on code-fix/docs/file-create/qa but **refactor α=0.300** — run 11's 0.865 at n=48 did not hold at n=96 |
| enhanced (production default, via PR #817 adapter) | **α=−0.837** — systematically anti-correlated on real trajectories (scripted run measured −0.679); worse than credulous (0.070) |
| evidence / rubric-heuristic | −0.038 both — activity-based judging collapse replicated |

Failure anatomy: annotator gold == verifier gold exactly, so every gemma miss is a genuine
judge error — it under-credits real completions whose final messages trail off mid-verification
or claim the edit failed (the workspace shows the work done). Same signal drives `enhanced`
below zero: claim-reading loses to workspace-verification on this distribution.

Consequences: (a) **no default flip** — the pre-registered gate is not met; gemma4:31b is
demoted from recommended-default pending either judge-prompt improvement for narration-free /
self-doubting claims or llama3.3:70b measured on these same trajectories (judging-only re-run,
cheap); (b) EVR-3 parity is directionally decisive on identical data: rubric-llm 0.694 vs
enhanced −0.837 (Δα=1.53) — `enhanced` should not remain the default evaluator on the merits,
but replacement waits for a gate-passing judge; (c) run-12 also surfaced a run-level infra
lesson: the first attempt produced 96/96 empty-response trajectories from a dropped profile
endpoint (fixed in PR #820) and was discarded as degenerate before judging completed.

### Runs 13–14 (2026-08-02) — replay judging: llama graduates; self-judging fails

Via the pack-replay executor on run 12's exact 96 trajectories (integrity clean both runs):
**run 13, llama3.3:70b: α=0.878 overall, refactor α=1.000 — PASSES** the gate against both
verifier and AI-annotator gold; combined with run 10 (scripted 1.000, n=96) llama holds every
graduation gate and is the pinned judge. gemma4:31b is demoted from the default calibrated set
(run 12: 0.694, refactor 0.300). **Run 14, qwen3-coder-tools:30b judging its OWN trajectories
fresh-context: α=0.469 (qa −0.033, refactor 0.100) — FAILS**: context isolation recovers 1.3 α
over in-loop self-assessment (−0.837) but weight-level correlation costs ~0.41 α vs llama on
identical data. Session-model self-judging cannot enter the calibrated set. Ladder + committed
reports: docs/architecture/judge-independence-experiments.md.

## Verdict and open items

> ⚠️ **Qualified by the SWE-bench-lite re-gate (2026-08-04, below).** Both gate-passing
> judges over-credit on the true shipping distribution (in-container-verified SWE-bench):
> gemma4:31b α=−0.52, llama3.3:70b α=0.26. Treat checklist item 2 (and the graduation
> recommendation) as **unproven on the production distribution** pending a balanced harvest.

**All three graduation gates are now cleared — by two complementary judges.** The scripted
gate at gating-grade n is met by llama3.3:70b (run 10: α=1.000, n=96), and the real-agent
trajectory gate is met by gemma4:31b (run 11: α=0.865, integrity clean, zero false
completions). `completion_strategy=rubric` has the evidence to graduate. Trust is
**judge-specific**: this graduates these calibrated judges, not LLM-judging in the abstract
(identical code scored 0.173–0.279 with other models, and one thinking model produced zero
usable grades — see point 4 and the run-8 guard).

Judge selection — **gemma4:31b is the recommended default**: it cleared BOTH the scripted
(0.929, run 7) and real-trajectory (0.865, run 11) gates, fits a 20 GB GPU fully (fast, no
offload), and is the model that carried the real-trajectory pass. llama3.3:70b is the
higher-accuracy alternative (1.000 scripted) where its ~42 GB footprint is affordable.

Graduation checklist for `completion_strategy=rubric` (TD-17 evidence):

1. ✅ **Confirmation at gating-grade n** — DONE (run 10, llama3.3:70b): `--variants 16`,
   n=16/family, 96/96 correct, α=1.000 every family. (Open: a variants-16 scripted
   confirmation for gemma4:31b specifically would let one judge own both gates; gemma4 has
   n=8 scripted (0.929) + real-trajectory (0.865) today.)
2. ✅ **Real agent trajectories** — DONE (run 11, gemma4:31b): α=0.865 on real
   qwen3-coder:30b trajectories, all families ≥ 0.7, zero false completions, integrity
   clean. The production distribution the gate exists for; the scripted-perfect `evidence`
   and `rubric-heuristic` baselines collapsed to −0.092 on the same trajectories.
3. ⏳ **Pin the judge identity in the flag criteria** — the remaining step: wire
   `completion_strategy=rubric` default-on ONLY with a calibrated judge (gemma4:31b or
   llama3.3:70b); an uncalibrated model or the heuristic fallback (α=−0.092) must revert to
   `enhanced`, per the ADR-011 fallback contract and the
   [flag-graduation policy](../../docs/architecture/flag-graduation-policy.md). This is a
   code change to the flag wiring, not another measurement.

Open follow-ups (no longer blocking graduation): a variants-16 scripted run for gemma4:31b
(single-judge rigor); the `--hard` corpus (run 0/PR #417) against the gate-passers to probe
discrimination past the α=1.0 ceiling; and the agent-side `edit` arg-format bug surfaced in
run 11 (the 30B model sends `ops` as a string).

### SWE-bench-lite real-distribution re-gate (2026-08-04) — the "real" gate wasn't real enough

Runs 11–14 treated the calibration-corpus real-agent trajectories as "the production
distribution." They are not: that corpus is 6 synthetic task templates, ~81% solved
(positive-heavy), with gold from the agent's own successes. To gate against the distribution
Victor actually ships into, we re-ran the judges on a **SWE-bench-lite** stratum — real GitHub
issues (astropy, django), the agent's `git diff` patch as the completing effect, and
**in-container FAIL_TO_PASS gold** (`--eval-backend docker`; an independent verifier, not the
agent's self-report). 30 instances, **5 resolved / 25 not** (17% solve — negative-heavy, the
inverse of the corpus). Stratum + per-judge results committed under `labels/swe-bench-lite-30/`;
produced by `victor.evaluation.swe_bench_stratum` + `gate_swe_bench_stratum.py`.

| Judge | corpus α (run 11/13) | **SWE-bench-lite α** | confusion (pos=5 / neg=25) |
|---|---|---|---|
| gemma4:31b | 0.865 (run 11) | **−0.523** | TP=5 FP=23 TN=2 FN=0 |
| llama3.3:70b | 0.878 (run 13) | **0.263** | TP=5 FP=10 TN=15 FN=0 |

Both collapse below the 0.7 gate. The failure mode is a **systematic positive bias**: perfect
recall (FN=0 — every real fix caught) but heavy false positives — gemma credits 28/30 complete
(near-constant "yes"), llama 15/30. On the positive-heavy corpus that bias mostly agrees with
gold and α looks high; on the negative-heavy real distribution it produces mass false
completions. **At real distribution an LLM completion judge cannot distinguish "looks done"
(plausible patch + confident summary) from "passes tests" — only the in-container verifier
can.** This confirms the effect-gate / acceptance-oracle thesis (ADR-012 / EVR) on real data,
and it *inverts* the run-11 story: there the activity baselines collapsed while the LLM judge
held; on true SWE-bench it is the LLM judge that collapses.

**Caveat (honest).** 5 positives is thin, so the α *point estimate* is noisy and the exact
llama > gemma ranking should not be over-read. What is robust at n=25 negatives is the
false-positive rate — gemma 23/25, llama 10/25 — either of which fails the gate on its own. A
balanced harvest (≥ ~20 positives, i.e. ~100–150 lite instances) would firm the α; the
qualitative conclusion (both judges over-credit on real negatives) does not depend on it.

**Consequence for graduation.** Checklist item 2 ("real agent trajectories — DONE") is met
only for the calibration corpus, not the shipping distribution. Do **not** graduate an LLM
completion judge (gemma4:31b or llama3.3:70b) as a default on this evidence; the in-container
verifier remains the trustworthy completion signal. Re-open item 2 pending a balanced
SWE-bench re-gate.

Reproduction::

    python benchmarks/judge_calibration/gate_swe_bench_stratum.py \
        ~/.victor/evaluations/eval_manifest_<id>.jsonl \
        --judge llama3.3:70b --endpoint http://<ollama-host>:11434 \
        --out benchmarks/judge_calibration/labels/swe-bench-lite-30/llama3_3.json

### Trained-classifier re-gate (2026-08-05) — the other half fails too

The LLM re-gate above showed LLM judges **over-credit** on the real distribution. The natural
follow-up: can a classifier *trained on real trajectories* do better? We harvested a second
30-instance run (13.3% solve) for a 60-instance pool (9 pos / 51 neg), then ran the honest
cross-run experiment — **train ModernBERT-base on run-1 (30 inst, 5 pos), test on run-2's
unseen instances (30 inst, 4 pos)** — no leakage. Result JSON: `labels/swe-bench-lite-60/`.

| Judge | real-SWE-bench α | failure mode |
|---|---|---|
| gemma4:31b (LLM) | −0.52 | over-credits (near-constant "complete") |
| llama3.3:70b (LLM) | 0.26 | over-credits (all solves caught, many FP) |
| **ModernBERT (trained on real)** | **−0.05** | **under-credits** — collapses to constant "incomplete" |

The classifier's held-out scores spanned only 0.017–0.117 (predicted-positive 0/30): with just
4 training positives it learned the majority class — TP=0, FP=0, TN=26, FN=4. So the two
substitute approaches fail in **opposite directions** — the LLM judge says "done" too often, the
small-data classifier says "not done" always — and **only the in-container verifier
discriminates.** This reproduces the run-12 encoder collapse (α=−0.27) with a clean cross-run
split, and closes the question the whole judge-independence arc opened: at the real shipping
distribution, the acceptance oracle (ADR-012 / EVR) is not merely preferable but **necessary** —
neither an LLM nor a cheaply-trained classifier is a viable stand-in until real positives are far
less scarce (a stronger agent, not a bigger judge). Model artifact not committed (598 MB);
regenerate via `benchmarks/judge_training/train_encoder.py` on the committed strata.
