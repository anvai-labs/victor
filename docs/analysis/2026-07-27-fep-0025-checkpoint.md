# FEP-0025 checkpoint — the loop runs, and nothing has earned promotion

Date: 2026-07-27
Status: paused at a clean boundary; nothing in flight

FEP-0025 Phases A–D are implemented and merged. The pipeline produces real
candidates from real traces, measures them against the shipped prompt on the
same tasks, and refuses the ones that do not clearly win. Three candidates have
been measured. All three were refused, and every refusal was correct.

The bottleneck is no longer the pipeline, the gates, the harness, or the
prompts. It is sample size.

## What shipped

| PR | What it fixed |
|---|---|
| #682 | `--provider all` evolved a hardcoded four that excluded the provider that ran |
| #687 | mutator rotation, per-call failover, reasoning-token budget, hygiene calibration |
| #691 | `task_type` reached its consumers instead of defaulting to `"default"` |
| #692 | suite artifacts persist per-task outcomes, so a contrast can be rebuilt from disk |
| #693 | code-generation benchmarks get source, not a git diff |
| #697 | `propose` — a human-written candidate can be measured, not just trusted |
| #704 | a throttled task is not evidence about the prompt |
| #705 | a lead must beat chance; an advisory is not a defect |

## The measurement that unblocked everything

MBPP had scored **0 on all 135 real tasks it had ever been given**, across eight
runs. `MBPPRunner` and `HumanEvalRunner` execute `agent_output + test_code` as
one Python file, so `agent_output` must be source; the benchmark callback
returned `trace.generated_patch`, a git diff. Line 3 of every `solution.py` read
`@@ -0,0 +1,27 @@` and the task died on `SyntaxError` before a test ran.

After the fix, and after excluding self-imports that the source capture was
concatenating out of `conftest.py`:

```
mbpp baseline: 0% -> 58.3%
```

Every prompt experiment before that fix was measuring nothing.

## Results

All three candidates target `GROUNDING_RULES` or `COMPLETION_GUIDANCE`, measured
paired against the shipped seed on identical tasks.

| candidate | origin | result | verdict |
|---|---|---|---|
| `45dfab10c582` | evolved | −2/24, disc 4/6, p=0.75 | refused |
| `785071738065` | evolved | +2/24, disc 8/6, p=0.79 | refused under the corrected gate |
| `b54762402bda` | human | −1/24, disc 4/5, p=1.00 | refused |

`785071738065` was *approved* before #705. Catching that mattered most: it was
the only result that could have shipped a change on evidence that did not
support it.

### Evolution learned something real

Given working failure traces for the first time, reflection proposed tool-usage
discipline that maps onto failures visible in the run logs: read ≥20 lines
before `edit()`, `old_str` copy-pasted verbatim with ≥3 lines of context, do not
retry an identical failing call, halt on a "closest match" path fallback. It
paid for the additions by compressing the SDLC mandates — 1911 chars against a
1912 seed. Under the pre-#687 `redundant_additions` rule that candidate would
have been rejected outright for rewording rather than appending.

### The human hypothesis worked and was neutralised

The naming rule ("the tests are the specification; match their identifiers, do
not edit the tests") cut its target failure class from **9 to 6** while net pass
rate moved −1. The intervention did what it was written to do; the gains were
eaten elsewhere ("other" failures rose 1 → 4).

At n=24 none of this is resolvable. Every candidate landed in the ±2, p≥0.75
band.

## Why n=24 cannot answer the question

Under the null the effect has mean zero and standard deviation
`sqrt(discordant)`. At n=24 we observe 9–14 discordant pairs, so the noise floor
is 3.0–3.7 and a candidate must reach roughly +4 to clear it. None did.

At n=60, expect 25–35 discordant, floor ≈5.5, and a genuine +6 becomes
detectable rather than invisible.

**This is the single highest-value next step.** Everything else built here is
waiting on it.

## Resume here

The paused run, unchanged (nothing had completed; no partial data recorded):

```bash
eval "$(victor auth env -p deepseek)"
victor benchmark run-prompt-suite mbpp \
  --prompt-section GROUNDING_RULES \
  --candidate-hash 785071738065 \
  --candidate-hash b54762402bda \
  --include-baseline -n 60 \
  --record-benchmark-results \
  --profile deepseek-v4flash-openai
```

Three arms share one baseline: 180 task-runs rather than 240, and the
head-to-head between the two candidates is exact. The evolved candidate runs
first so the priority result survives if quota gives out. Roughly 3 hours.

`--record-benchmark-results` defaults to **off**. Without it the arms run and
nothing syncs — no verdict, no evidence recorded.

### Provider state as of pausing

| provider | state |
|---|---|
| zai / glm-5.2 | 429. Every one of the 14 evolve-run 429s today was zai. Fine as a rotation mutator, hopeless for a 48+ task benchmark |
| moonshot / kimi-k3 | 429, exhausted after ~72 tasks |
| deepseek / deepseek-v4-flash | healthy. ~57s/task. Faster than `-pro` (5.4s vs 7.7s on a code task) and does not burn budget on hidden reasoning |

Both arms of a comparison must use the same provider. Cross-run comparisons are
not valid: the kimi baseline (58.3%) and the deepseek baseline (58.3%) coincide
by chance, and #695 pinned kimi to temperature 1.0 mid-session.

## Known and deliberately unfixed

**The naming failure class is still 9/24 of baseline failures** — the largest
remaining, and now known to be *partially* addressable by instruction. Worth a
second attempt with wording that does not suppress the `conftest` shim workaround
that was converting some failures into passes.

**PrefPO appends generic restatements.** Seen on two separate candidates:
`- Read the error message carefully.` and `- Check command syntax.`, both
restating the paragraph above them. Lexical similarity cannot catch it —
measured at 0.36 Jaccard against the line it restates, and a strict-subset rule
finds it too (the reworded line drops the word "carefully"). The redundancy is
semantic. The fix is likely to stop PrefPO appending category hints to sections
that already cover them, not another similarity heuristic.

**The candidate store is fine.** All 13 candidates are HOLD — legitimate and
under-measured, not corrupt. `purge` correctly reports nothing to delete. Before
#705 it would have deleted five of them including the best one.

## Traps worth remembering

- **`--record-benchmark-results` is off by default.** One full paired run was
  spent before noticing.
- **Sequential arms on one provider exhaust quota, and the last arm pays.** A
  third arm recorded 0/24 in 128 seconds against a prompt that never ran. #704
  stops that becoming a permanent "failed benchmark" record, but it does not
  create quota.
- **ENOSPC corrupts runs silently.** It hit one task mid-arm. Rust `target/`
  directories across proximaDB worktrees reached 165 GB; deleting them recovered
  the space without touching source. Concurrent sessions regrow them.
- **Commit onto a branch cut from `develop`, not whatever is checked out.** Three
  commits this session landed on branches that were merged out from under them.
  All recovered from reflog, but concurrent sessions make it likely.
