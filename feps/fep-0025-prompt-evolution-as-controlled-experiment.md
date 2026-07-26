---
fep: "0025"
title: "Prompt evolution as a controlled experiment — seeding, evidence, and promotion into source"
type: Standards Track
status: Draft
created: 2026-07-26
modified: 2026-07-26
authors:
  - name: Vijaykumar Singh
    email: singhvjd@gmail.com
    github: vjsingh1984
reviewers: []
discussion: https://github.com/anvai-labs/victor/discussions
---

# FEP-0025: Prompt evolution as a controlled experiment

## Table of Contents

1. [Summary](#summary)
2. [Motivation](#motivation)
3. [Proposed Change](#proposed-change)
4. [Benefits](#benefits)
5. [Drawbacks and Alternatives](#drawbacks-and-alternatives)
6. [Unresolved Questions](#unresolved-questions)
7. [Implementation Plan](#implementation-plan)
8. [Migration Path](#migration-path)
9. [Compatibility](#compatibility)
10. [References](#references)
11. [Acceptance Criteria](#acceptance-criteria)

---

## Summary

Prompt evolution is built as **generate-and-observe**. Making it work requires
**generate-and-compare**. Every remaining defect follows from that one framing
error, and no amount of further plumbing repair will fix it.

FEP-0017 closed the serve/reward loop, and a series of 2026-07-25/26 fixes made
the evidence trustworthy: verdicts are joinable, corruption is gated, run kind
and provider are recorded rather than inferred. With all of that landed, an
overnight run produced a candidate scored `mean 0.92` from 51 real outcomes —
whose entire diff was the removal of six blank lines.

That is the system working exactly as designed and still producing nothing. This
FEP restructures it around the experiment, not the candidate:

- the **shipped prompt becomes an arm**, so "better" has a referent;
- **benchmark runs become the experiment driver**, which collapses the seeding
  circularity rather than working around it;
- **reflection receives failing evidence** instead of a histogram of category
  counts against a prompt it can only partly read;
- **promotion into `prompt_section_texts.py`** becomes a reviewed, gated step,
  so improvements reach other installs instead of dying in one laptop's SQLite.

## Motivation

Each claim below was measured on this repository between 2026-07-25 and
2026-07-26, not inferred. See
`docs/analysis/2026-07-25-prompt-evolution-audit.md` for the full trail.

### G1 — There is no control arm, so "better" is undefined

`BASELINE_CANDIDATE_HASH = "__baseline__"` exists, but only
`benchmark run-prompt-suite` ever uses it — a manual, hand-invoked A/B. The live
loop selects among *candidates*; the shipped text is never an arm. Thompson
sampling therefore answers "which candidate looks best" and never "is any of
this better than what we already ship."

### G2 — The reward is non-comparative, and its headline number is inherited

The overnight zai candidate reported `mean 0.92`, `alpha=49`, `beta=4`,
`sample_count=51`. None of that came from comparing it to anything: the
posterior is seeded from the provider's historical success rate at creation
time. A candidate identical to the shipped text would report the same 0.92.

Per-turn `completion_score` also remains a confounded proxy for interactive
sessions — the harness verdict now wins wherever one exists (2026-07-26), but
nothing grades interactive work.

### G3 — Reflection is a histogram, and it reads a truncated prompt

`GEPAService.reflect()` receives `traces_summary` — aggregated failure-category
frequencies — plus `current_text[:1000]`. It never sees a failing error message,
a tool-call sequence, or a diff. And **four of the seven evolvable sections
exceed 1000 characters**:

| Section | chars | reflected on |
|---|---|---|
| `ASI_TOOL_EFFECTIVENESS_GUIDANCE` | 2934 | first 34% |
| `GROUNDING_RULES` | 1912 | first 52% |
| `COMPLETION_GUIDANCE` | 1551 | first 64% |
| `GROUNDING_RULES_EXTENDED` | 1066 | first 94% |

The mutator is asked to repair failures it cannot see, in text it has only
partly read. It responds the only sensible way: by returning approximately its
input. `COMPLETION_GUIDANCE` produced whitespace-only collapse on two
independent runs.

### G4 — The keying dimension is decoration

Candidates are keyed `(section, provider)`. After the overnight run, hash
`a88e8523` existed under **five providers** — `openai`, `xai`, `deepseek`,
`anthropic`, `zai` — with byte-identical text. Provider-specific evolution
produces no provider-specific prompts. Meanwhile the dimension that plausibly
does carry variance — failure mode, task type — is not a key at all
(`task_type` is `default` for every trace).

Worse, `--provider all` iterated a hardcoded `["openai", "xai", "deepseek",
"anthropic"]`, so the provider that actually ran all night could not be evolved
while four that never ran were, from its traces.

### G5 — Seeding is circular

`seed_from_evaluations()` credits a candidate only if the artifact names it. An
artifact names a candidate only if one was *served*. Nothing is served until a
candidate exists and is selected. The frontier therefore stays empty
(`agent_prompt_pareto_instance` = 0 rows) even after a night that produced 123
graded tasks. Stamping identity after the fact (2026-07-26) narrowed this but
did not break the cycle: the run must already know its arm.

### G6 — Nothing reaches other users

Evolution output lives in `~/.victor/victor.db`. There is no supported route
into `victor/agent/prompt_section_texts.py`, so even a genuinely better prompt
benefits exactly one machine. `scripts/prompt_candidates.py` was added as a
manual bridge; it is a stopgap, not a mechanism.

## Proposed Change

### The unit of work becomes the experiment

```text
PromptExperiment
  section        COMPLETION_GUIDANCE
  arms           [ baseline(shipped) , variant_a , variant_b ]
  assignment     randomized per task, recorded before execution
  population     {run_kind, benchmark, task_type}
  stopping rule  min_n per arm AND effect size outside CI
  outcome        per-arm verdict distribution
```

Three consequences, each removing a gap rather than mitigating it:

- **Baseline is always an arm** (G1). Every experiment measures a difference.
- **Assignment precedes execution** (G5). The arm id is chosen when the task
  starts, so the artifact records it by construction — no post-hoc stamping, no
  circularity.
- **Reward is a contrast, not a level** (G2). The reported statistic is
  `p(variant) − p(baseline)` with an interval, computed from the harness verdict.

### Benchmarks become the experiment driver

Evaluation runs are the only population carrying ground truth, and they are
uniform, repeatable and diverse — the properties an experiment needs. The
benchmark harness gains an experiment hook:

```text
for task in tasks:
    arm = experiment.assign(task)          # randomized, recorded
    result = run(task, prompt=arm.text)
    experiment.observe(arm, result.status, result.tests_passed/total)
```

Interactive sessions continue to serve candidates under the FEP-0017 epsilon
path, but their outcomes are recorded as **observational**, kept separate from
experimental arms and never used for promotion. That preserves the
already-implemented loop while refusing to let an ungraded proxy decide what
ships.

### Reflection receives evidence

`reflect()` changes shape:

- **No prompt truncation.** Pass the full section; the 1000-char cap predates
  sections three times that size.
- **Failing exemplars, not counts.** For each of the *k* worst tasks in the
  losing arm: the task id, the error text, the tool-call sequence, and the diff
  the agent produced. A category histogram cannot tell a mutator what to write;
  a failing transcript can.
- **Contrastive framing.** Where an experiment has both arms, show the mutator
  where baseline succeeded and the variant failed. That is the signal GEPA's
  reflective step is designed around and currently never receives.

### Keying follows variance, not org chart

Key candidates by `(section, population)` where population is
`{task_type, failure_mode}`, with provider as a *filter* on eligible evidence
rather than part of the key. This stops five identical texts occupying five rows
and lets a variant target "tasks that failed with `edit_mismatch`" — a class the
prompt can actually address.

Prerequisite: `task_type` must be populated (it is `default` for every trace
today because nothing emits `task_classification`).

### Promotion into source is the terminal step

A variant that wins its experiment becomes a **pull request**, not a database
row:

```text
victor prompts promote <experiment-id>
  -> writes prompt_section_texts.py with the winning text
  -> embeds provenance: experiment id, arms, n, effect size, CI
  -> regenerates the section digest
  -> opens a PR for human review
```

Gate: `n >= min_n per arm`, effect size positive with the interval excluding
zero, and the corruption checks (`truncated_tail`, `redundant_additions`)
passing. Promotion is deliberately human-reviewed — the repo is the artifact,
the DB is only evidence.

## Benefits

- **"Better" becomes a measurable claim.** Today no number in the system
  answers it; after this, one does.
- **The seeding circularity disappears** rather than being worked around.
- **Mutation gets the input it was designed for**, which is the difference
  between whitespace collapse and a real variant.
- **Improvements reach other installs**, which is the point of evolving prompts
  in a shipped product.
- **Negative results become cheap and legible** — an experiment that shows no
  effect is a finished experiment, not an inert row.

## Drawbacks and Alternatives

- **Cost.** Randomized arms mean part of every benchmark runs the control. That
  is the price of knowing; the alternative is the current state, where a whole
  night of tokens yielded no comparison. Mitigate with baseline-sharing across
  concurrent experiments on different sections.
- **Slower verdicts.** Effect sizes on pass/fail need real *n*. Small effects may
  need more tasks than one night. Prefer few, well-powered experiments over many
  underpowered ones — the current design has effectively infinite underpowered
  ones.
- **Alternative — keep Thompson sampling, add a baseline arm only.** Cheaper,
  and worth doing first (Phase 1). Insufficient alone: it fixes G1 but leaves
  reflection blind (G3) so the arms stay near-identical and the contrast is
  always zero.
- **Alternative — offline scoring of variants against recorded traces.** No live
  cost, but it cannot measure a prompt's effect on behaviour, only its agreement
  with past behaviour. Rejected as the primary mechanism; useful as a pre-filter
  to cull obviously-bad variants before spending tasks on them.

## Unresolved Questions

- **What is the minimum viable *n*?** At the observed mbpp rate (18/123 ≈ 15%
  pass), detecting a 5-point absolute improvement needs a few hundred tasks per
  arm. Either accept coarser effects, choose benchmarks with higher baseline
  variance, or use per-task paired assignment to cut variance.
- **Do sections interact?** Arms are per-section, but the prompt is assembled
  from all of them. Two simultaneous experiments may confound. Sequential
  experiments are safe but slow; factorial designs need more thought.
- **What counts as the outcome for interactive sessions?** Left observational
  here. `/rate` and completion markers are candidates but both are sparse.
- **Provider generalization.** If a variant wins on zai, does it ship for
  everyone? Proposal: promote to source only on evidence from ≥2 providers, else
  keep it provider-scoped.

## Implementation Plan

Phased so each lands independently and is separately revertible.

| Phase | Change | Closes |
|---|---|---|
| 1 | Baseline as a first-class arm in the live selector; report contrast not level | G1, G2 |
| 2 | `reflect()` takes full text + k failing exemplars; drop the 1000-char cap | G3 |
| 3 | Experiment assignment in the benchmark harness; arm recorded pre-execution | G5 |
| 4 | Re-key on `(section, population)`; emit `task_classification` so `task_type` is real | G4 |
| 5 | `victor prompts promote` → codegen + provenance + PR | G6 |

Phase 2 is the highest value-per-line: it is a prompt-construction change, and
it is the direct cause of the whitespace-only mutations observed twice.

## Migration Path

Additive. Existing candidates keep working under the current selector until
Phase 1 flips the default; the `agent_prompt_candidate` table gains
`experiment_id` and `arm` columns (nullable) rather than being replaced. Rows
without an experiment are treated as observational and are never promotable —
which correctly describes every row that exists today.

## Compatibility

No wire-contract change. `prompt_section_texts.py` remains the single source of
truth for shipped text; this FEP only adds a reviewed path for changing it.
`exploration_enabled=false` continues to disable live serving entirely.

## References

- FEP-0017 — closed the serve/reward loop (Implemented)
- `docs/analysis/2026-07-25-prompt-evolution-audit.md` — the measurements cited
- PRs #664, #666, #667, #670, #672, #675, #677 — the evidence-plane repairs this
  FEP builds on

## Acceptance Criteria

1. A named experiment can be created for a section with baseline + ≥1 variant.
2. A benchmark run assigns arms randomly and records the arm in each artifact
   **before** the task executes.
3. The system reports per-arm pass rate, effect size and interval.
4. A variant failing to beat baseline is closed as such, and is never served on
   merit.
5. A winning variant produces a reviewable PR against
   `prompt_section_texts.py` carrying its provenance.
6. Reflection input for a >1000-char section contains the full text and at least
   one failing exemplar, demonstrated by test.
