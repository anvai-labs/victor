# Prompt-evolution audit — 2026-07-25

Audit of the three GEPA candidates in `~/.victor/victor.db` produced by
`/prompt-optimize` on 2026-07-25, and of the machinery that produced them.

**Verdict: nothing in this batch is promotable.** All three candidates were
purged. The defects they exposed are fixed in code, so the *next* batch is
worth auditing on its merits.

## What the CLI table reported

```text
CONCISE_MODE_GUIDANCE            moonshot  1  Evolved    +82 chars  e6bcc0c0 -> 949292ba
COMPLETION_GUIDANCE              moonshot  1  Evolved    -55 chars  c9749226 -> ffb30907
(7 other sections)                                       No change
```

The table is accurate but easy to over-read: **"Evolved" means "a candidate row
was written"**, not "the prompt improved" and not "the change is live". It also
filters to the session's provider, hiding a third candidate created five
minutes earlier under `zai`.

## What was actually in the database

| Section | Provider | Hash | Samples | Benchmark | Servable? |
|---|---|---|---|---|---|
| `ASI_TOOL_EFFECTIVENESS_GUIDANCE` | zai | `e906e4a6` | 0 | 0.05 over 4 runs, 0 passed | **yes** |
| `CONCISE_MODE_GUIDANCE` | moonshot | `949292ba` | 0 | never run | no |
| `COMPLETION_GUIDANCE` | moonshot | `ffb30907` | 0 | never run | no |

Every row: `sample_count=0`, `completion_score=0`, `is_active=0`,
`instance_scores={}`, `coverage_count=0`. `agent_prompt_pareto_instance` held
**zero rows** — the Pareto frontier the selector consults has never had an
instance scored into it.

So: three mutations were generated and stored; none was ever measured.

### Did prompt optimization "do its work"?

It did the *generation* half. The **selection** half never ran, which is the
half that makes it optimization rather than mutation. This matches
[FEP-0017](../../feps/fep-0017-prompt-optimization-reward-loop.md), still Draft:
candidates are created but never served, rewarded, or improved. This audit
confirms the loop is still open as of 2026-07-25 and adds two defects FEP-0017
does not cover (provenance and corruption, below).

## The three candidates

### `COMPLETION_GUIDANCE` — corrupt, rejected

The candidate ends mid-sentence:

```diff
-- Read error messages carefully and check command syntax before reporting a blocker.
+- Read error messages carefully and
```

**Root cause, confirmed.** `max_prompt_chars` defaults to `1500`. The shipped
`COMPLETION_GUIDANCE` baseline is **1551 chars — already over the cap**. Every
mutation of it was therefore truncated, and `boundary_aware_truncate` cut at the
last *space* before 1500, landing mid-clause. The stored candidate is exactly
1496 chars. The cap, not the strategy, decided this candidate's content.

The remaining `-55` is blank-line removal (six blank lines collapsed).

### `CONCISE_MODE_GUIDANCE` — degenerate, rejected

Purely additive; the two added lines are the whole diff:

```diff
 - Check command syntax before reporting a blocker.
+- Read the error message carefully.
+- Verify file paths with ls() before reading.
```

The first restates the line two above it ("Read error messages carefully before
retrying."). The second is tool-discipline guidance in a section headed **OUTPUT
STYLE: CONCISE** — and it is telling a *concise-output* section to grow. This is
degenerate growth, not evolution.

### `ASI_TOOL_EFFECTIVENESS_GUIDANCE` (zai) — unproven and mis-seeded, rejected

The most interesting of the three: a genuine 13-rule → 8-rule compression
(2477 → 1029 chars) and, read on its own, decent prose. It still fails:

- **Benchmark 0.05 across 4 runs, 0 passed.** It is the only candidate with
  measurement, and the measurement is bad.
- **Parent hash `0188a1b6cf72` does not match the shipped baseline.** It was
  evolved from a seed that is not what ships, so its diff is not a diff against
  production.
- It drops rule 13 (`code`/`git` are not shells) and rule 4 (project scope) —
  the two rules most load-bearing for tool-call correctness.
- **It was servable.** `requires_benchmark=0` for its strategy chain, so
  `_servable_candidates` admitted it and Thompson sampling could inject a
  benchmark-failing prompt into live `zai` sessions.

## Is the evidence behind this valid?

Mostly not — though the *source* is right and the failures are all fixable
plumbing, independent of the candidates themselves.

**Traces are global, and the audit trail says so.** The pool is
`~/.victor/logs/usage.jsonl` (+ rotated `.gz`) — `global_logs_dir`, shared by
every project on the machine. The current file holds 44 sessions across
several repos, with no project field captured anywhere in the aggregation.

**A large minority of the evidence is eval harness — the right source, used
wrongly.** Classifying sessions by their prompts and weighting by the signal
that actually drives reflection:

| Session kind | `tool_result` events | Share |
|---|---|---|
| eval harness | 240 | **31%** |
| interactive | 538 | 69% |

> **Corrected 2026-07-25.** An earlier revision of this document reported a 59%
> eval share, counting any session carrying *"WARNING: N turns remaining out of
> 10"* as a benchmark run. That turn budget is a shared mechanism: inspecting
> the sessions directly shows most of them are **delegate/subagent runs on this
> user's own repositories** ("Fix Bug 2: Reconcile `AnalysisFulfillment.check()`
> …", "Review the single Rust crate …"), not benchmarks. Counting only the
> unambiguous SWE-bench prompt (*"Fix the following issue by editing the source
> code in this repository"*) gives 31%. The direction of the argument is
> unchanged; the magnitude was overstated.

Evals are the population a harness-level prompt *should* be tuned on: uniform
tasks, diverse problems, repeatable, and — uniquely — carrying a ground-truth
outcome. Interactive sessions have no verifiable success label at all. A
substantial eval share is a feature.

The defect is that **the pipeline consumes the eval population and discards the
eval signal.** `completion_score = 1 - failure_rate * 1.5` is computed purely
from tool-call failure rate; whether the task actually resolved is never read.
So the one population that comes with a real label is scored by a proxy that
correlates with *tidy tool use*, not with *solving the problem*.

That is not hypothetical — the two disagree inside a single artifact:

```json
{"task_id": "task-4", "benchmark": "dr3_eval",
 "status": "failed", "completion_score": "1.0"}
```

The run failed. The proxy scored it perfect.

**Now fixed** (see *Fixes landed*). Where a benchmark graded the session, its
verdict wins; the proxy survives only for interactive sessions, which nothing
grades, and `ExecutionTrace.score_source` says which was used. Note that the
verdict is derived from `status` and the test counts — deliberately *not* from
the artifact's own `completion_score`, since that is the field shown
contradicting itself above.

### The eval-driven path works — it just is not wired to ordinary eval runs

`seed_from_evaluations()` reads `~/.victor/evaluations/eval_*.json` and writes
per-instance pass/fail into the Pareto frontier that `recommend()` selects from.
`_artifact_identity()` requires each artifact to record *which candidate was
under test*. Across the 5,464 artifacts on disk:

| | count |
|---|---|
| total `eval_*.json` | 5,464 |
| …synthetic (`model` = `test` / `test-model`) | 4,773 |
| …real-model runs | ~687 |
| stamped with a candidate hash | 107 |
| …of which are the `cand-123` test fixture | 100 |
| **real, stamped eval runs** | **7** |

Benchmarks present: `human_eval` 4,417, `guide` 422, `mbpp` 320, `dr3_eval` 265,
`swe_bench` 40.

Those 7 real stamped runs are all `glm-5.2`: 4 against candidate
`e906e4a6e104` and 3 against `__baseline__`. **That is exactly the
`benchmark_runs=4, benchmark_score=0.05` on the ASI candidate above** — so the
eval → candidate → score link is real and end-to-end functional. It is the only
genuine measurement anywhere in this system, and it came from an eval.

The gap is that stamping is *manual*: it happens only when someone runs
`victor benchmark --prompt-candidate-hash <hash> --prompt-section-name <section>`
as a deliberate A/B. The ~680 other real-model runs across `human_eval`, `mbpp`,
`swe_bench`, and `guide` carry no candidate identity, so none of them feed
selection — and `agent_prompt_pareto_instance` stays at zero rows.

**This was the highest-leverage gap, and it is now closed** (see *Fixes landed*
below). Artifacts carry a new `observed_prompt_identities` field recording what
the runtime actually served, so every eval run — not just a hand-configured A/B
— becomes ground-truthed selection evidence.

Demonstrated on a real unstamped artifact
(`eval_swe_bench_20260413_174321.json`, 10 tasks):

```text
before: 0 usable instance scores
after:  10 instance scores (1 passed, 9 failed) attributed to the served candidate
```

Two further consequences of *mixing* rather than of evals themselves:

- **The pool is unlabelled**, so eval and interactive sessions evolve one prompt
  jointly. Either population alone is interpretable; the blend is not — and the
  harness-specific turn pressure ("make your edits NOW") is scaffold artifact,
  not task signal.
- **Lessons are not routed to the section they belong to.** The
  `CONCISE_MODE_GUIDANCE` mutation appended "Read the error message carefully"
  and "Verify file paths with ls() before reading" — plausible lessons from a
  tool-failure-heavy population, grafted onto a section headed **OUTPUT STYLE:
  CONCISE**. Right lesson, wrong section.

**Provider attribution was silently broken.** Both collectors
(`_collect_traces`, `_collect_traces_v2`) initialised `sessions[sid]["provider"]
= ""` and **never assigned it**, so every `ExecutionTrace` reported
`provider="unknown"` — even though `session_start` / `stream_completed` events
carry the real value. Meanwhile `evolve()` persisted the candidate under the
*current session's* provider. A candidate labelled `moonshot` was therefore
reflected from a pool that is 41% ZAI, 14% Ollama, and 2% DeepSeek.

Measured on the live log after the fix:

```text
before: unknown 44/44
after:  zai 18, moonshot 18, ollama 6, deepseek 1, unknown 1
```

**`task_type` is always `default`.** The aggregation reads it from
`task_classification` events; the current log contains none.

## Should any of this be sourced into code?

**No — none of the three.** One is truncated, one is degenerate, and the third
failed its benchmark and was seeded from stale text. Promoting any of them would
put unmeasured (in one case measurably worse) text into everyone's install.

The underlying instinct is right, though, and worth stating as policy:

> `~/.victor/victor.db` is one laptop's scratchpad. Anything that should reach
> other users belongs in `victor/agent/prompt_section_texts.py` — version
> controlled, reviewed, diffable, and installed with the package. The DB is
> *evidence*; the repo is the *artifact*.

`scripts/prompt_candidates.py` is that bridge: `audit` classifies every
candidate against the shipped baseline, `show` diffs one, `export` emits a
paste-ready literal (and refuses anything not classified `PROMOTE` without
`--force`), and `purge` drops rejects after backing the table up.

## Fixes landed with this audit

| Defect | Fix |
|---|---|
| Truncation cut mid-sentence | `boundary_aware_truncate` now cuts at line → sentence → word boundary, dropping the partial trailing line |
| Bloat cap below the shipped baseline amputated every mutation | effective cap floored at `len(current_text)`; a bloat control cannot sit under the text it measures growth from |
| Truncated / degenerate candidates reached the DB | new `truncated_tail` and `redundant_additions` hygiene violations, both enforced at the persist gate |
| Every trace reported `provider="unknown"` | `_absorb_session_identity` fills provider/model from any event carrying them; `_normalize_provider_label` maps `MoonshotCompatProvider` → `moonshot`, `SandhiOllamaProvider` → `ollama` |
| Provider-scoped candidates reflected from a mixed pool | `_scope_traces_to_provider` filters the pool to the provider being evolved, falling back (with a log) when it is too small |
| Eval artifacts named a candidate only under an explicit `--prompt-candidate-hash` A/B, so ~680 real runs fed nothing | the eval adapter samples served identities per task, the harness unions them into `observed_prompt_identities`, and `_artifact_identities()` reads them — so every eval run becomes Pareto evidence |
| `completion_score` was a tool-failure proxy even for graded sessions, and three collectors each used a *different* proxy | `session_id` is now serialized per task (the join key `TaskResult` already held in memory), `_harness_verdicts()` maps session → verdict, and all three collectors score through one `_score_session()`: harness verdict where a benchmark graded the session, proxy only where nothing did. `ExecutionTrace.score_source` records which |
| Run kind had to be inferred from prompt text, and the shared turn-budget notice made delegate work read as benchmark runs | every usage event now carries `run_kind` (`interactive`/`evaluation`/`delegate`/`headless`) stamped by whoever starts the run — a `ContextVar` scope so nesting and concurrency behave — and it reaches `ExecutionTrace.run_kind` |

Verified against the real candidates: the persist gate now flags
`redundant_additions` on the CONCISE mutation and `truncated_tail` on the
COMPLETION mutation, while the ASI full rewrite trips only non-enforced signals
— legitimate rewrites still pass.

### One prompt change *was* sourced into core — by hand

`ASI_TOOL_EFFECTIVENESS_GUIDANCE` rules 4 and 13 were rewritten directly in
`prompt_section_texts.py`, from evidence in the same session that produced this
audit rather than from a GEPA candidate:

- **Rule 13** ended `anything else → shell(cmd='...', action='exec')`. A model
  copied the literal `'...'`; `/bin/sh` replied `...: command not found`, and
  the session concluded the shell tool did not exist. A placeholder inside
  guidance is an instruction to emit that placeholder. It now shows a real
  command, and `shell` names the mistake instead of executing it.
- **Rule 4** read "Stay in project scope… unless the task explicitly requires
  external paths" with no route for that exception. The same session read that
  as *unreachable* and abandoned a cross-repo task, even though `ls()` and
  `shell()` are not scoped at all (only `read()` is). It now states the routes.

This is the promotion path working as intended: evidence from traces → a
reviewed diff against version-controlled prompt text → shipped to every
install. The mechanism is the same whether the proposal comes from GEPA or from
reading a transcript; only the source of the hypothesis differs.

## Still open

- **The loop itself** (FEP-0017): candidates are still never served, rewarded,
  or promoted to `is_active`. Until that lands, `/prompt-optimize` generates
  without selecting.
- **`requires_benchmark=0` plus no benchmark run means servable-unmeasured.**
  The ASI candidate could have been injected on 0.05/4. The serve gate should
  require *some* evidence, not merely the absence of a benchmark requirement.
- **No project scoping on traces.** Provider scoping landed here; project
  scoping needs a project field at the emission site first.
- **Backfill was investigated and is not recoverable** — see below. Only runs
  from here on carry identity.
- **Scope evolution by run kind.** Events now carry `run_kind` (see *Fixes
  landed*) and it reaches `ExecutionTrace`, but nothing filters on it yet: a
  section could be evolved on evaluations alone, on interactive alone, or on
  both deliberately. Sessions logged before the tag existed stay `unknown` and
  must not be back-inferred — that inference is what produced the error
  corrected above.
- **Sections are not topic-fenced.** Tool-discipline lessons landed in an
  output-style section. Reflection should know which section it is mutating and
  reject guidance that does not belong to it.
- **Reward is a tool-failure proxy.** Real task outcomes (`/rate`, completion
  markers) should feed `completion_score`.
- **The cap is still section-blind.** `COMPLETION_GUIDANCE` ships 51 chars over
  the default; flooring at the seed length works around that rather than
  reconciling per-section budgets.

## Why historical artifacts cannot be backfilled

Serializing `session_id` fixes attribution going forward. It was worth asking
whether the ~1,600 existing tasks could be joined retroactively. They cannot,
and the reason is worth recording so nobody repeats the search.

**The sessions were never durable.** Each task runs inside a
`tempfile.mkdtemp()` workspace that the agent chdirs into (`harness.py`,
`agent_adapter.py`), while `ConversationStore` derives its path from the current
working directory. Every eval task's conversation store lived in a directory
deleted moments later. The data was not misfiled — it was never persisted.
(Fixed: see `durable_evaluation_conversations`.)

**No join key of sufficient specificity survives.** Restricting to tasks where
an agent actually ran leaves 889 tasks across 48 runs:

| | runs |
|---|---|
| no candidate session exists at all | 18 |
| fewer candidate sessions than tasks (e.g. 50 tasks, 1 candidate) | 27 |
| enough sessions to be 1:1 | 3 |

Every candidate session's `project_path` is the repository, never a temp
workspace — they are ordinary interactive sessions that overlapped in time.

**Both fallbacks fail too.** The global `usage.jsonl` survives temp workspaces,
but covers only 2026-07-08 onward while the runs span 2026-04-13 to 2026-07-15;
7 of 48 runs fall inside coverage, 5 with zero matching sessions and 2 with a
single candidate for 20 tasks each. Matching on message content instead of time
finds **zero** sessions naming any of the 76 specific benchmark task ids.

Every available join is many-tasks-to-one-session. That does not recover a lost
fact, it invents one — and it would inject a fabricated ground-truth score into
the reward path. A missing verdict correctly falls back to the labelled proxy;
a wrong verdict does not.

## Reproducing

```bash
python scripts/prune_eval_corpus.py report         # corpus composition
python scripts/prompt_candidates.py audit
python scripts/prompt_candidates.py show <hash>
python scripts/prompt_candidates.py purge          # dry run
python scripts/prompt_candidates.py purge --apply  # backs up, then deletes
```

The 2026-07-25 batch was purged to backup table
`agent_prompt_candidate_backup_20260725_121347`.

The evaluation corpus was 5,565 artifacts of which **48 (1%)** reflected an
agent actually working — 88% were test fixtures (`model: test`/`test-model`) and
12% recorded no tool call at all. Counting them together is what made the
figures above hard to read. The noise is archived (not deleted) under
`~/.victor/evaluations/archive/`, reversible with
`prune_eval_corpus.py undo --apply`.
