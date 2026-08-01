# ADR-027: Prompt-Optimization Strategy Fidelity and Honest Naming

## Metadata

- **Status**: Accepted
- **Date**: 2026-08-01
- **Decision Makers**: Vijaykumar Singh
- **Related ADRs**: 009 (rubric completion), 012 (regression-gated harness acceptance)
- **Governed by**: [FEP-0017](../../../feps/fep-0017-prompt-optimization-reward-loop.md)
  (reward loop), [FEP-0025](../../../feps/fep-0025-prompt-evolution-as-controlled-experiment.md)
  (controlled experiment)
- **Scope**: `victor/framework/rl/learners/strategies/` only. No change to the
  `PromptOptimizationStrategy` protocol, the config keys, or the reward loop.

## Context

Victor's prompt-optimization layer selects, per section, a chain of strategies
named after the literature: `gepa`, `miprov2`, `cot_distillation`, `prefpo`
(`victor/config/prompt_optimization_settings.py`). Only **GEPA** is
LLM-driven. The other three are heuristic adaptations — legitimately so — but
two problems had accreted:

1. **Name-vs-reality drift.** The class names and the module docstring claimed
   more than the code did.
   - `MIPROv2Strategy` is a query-aware **KNN few-shot retriever**
     (`select_similar_traces`), not MIPROv2's instruction/Bayesian proposal
     search. Its own class docstring already said so; the package docstring did
     not.
   - `CoTDistillationStrategy._distill_reasoning` emitted a **fixed**
     discover→read→plan→edit→verify template with two `if` branches. It never
     read the source trace's reasoning or tool sequence, so the header
     "distilled from a 94%-scoring trace" was decoration. This is the worst
     offender: the name asserts a behaviour the code did not perform.
   - `PrefPOStrategy` is honestly documented as deterministic, but see (2).

2. **PrefPO cross-section contamination.** PrefPO ranked failure categories
   *globally* and appended the top failure's hint to whatever section it was
   optimizing. So an output-style section (`CONCISE_MODE_GUIDANCE`) could
   receive tool-discipline guidance ("copy `old_str` exactly") for an
   `edit_mismatch` unrelated to verbosity. The 2026-07-27 FEP-0025 checkpoint
   recorded exactly this ("tool-discipline guidance landed in the output-style
   section") as a known, deliberately-unfixed defect.

The config keys (`miprov2`, `cot_distillation`, `prefpo`) are load-bearing:
they appear in `prompt_section_registry.py` section defaults,
`optimization_injector.py`, and stored candidate `strategy_name` /
`strategy_chain` rows. Renaming them is a breaking change to on-disk evidence.

## Decision

**Make the strategies faithful to their names where cheap, tell the truth in
the docs where a rename is not, and keep the config keys stable.**

1. **CoT distillation becomes faithful.** `_distill_reasoning` now derives the
   scaffold from the source trace's **real trajectory**: the ordered sequence
   of *successful* tool calls (`tool_call_details`), collapsing consecutive
   identical tools into one step and carrying each call's recorded
   `reasoning_before` into the step text. A recovery step is appended only for
   a failure the trace actually hit (`edit_mismatch`). The prior fixed template
   is retained **solely** as a fallback for traces captured before ASI
   tool-call detail existed, and the header states which basis was used
   ("an observed trace" vs "the success profile"). The "distilled from" claim
   is now true.

2. **PrefPO becomes section-scoped.** A `SECTION_RELEVANT_FAILURES` map defines
   which failure categories each target section may address; failures outside a
   section's concern are dropped before ranking. `CONCISE_MODE_GUIDANCE` is
   scoped to `verbosity`; `GROUNDING_RULES` and `COMPLETION_GUIDANCE` to their
   respective tool-discipline / execution categories. A section absent from the
   map stays unscoped, preserving behaviour for custom sections. This closes
   the checkpoint's contamination defect at its source.

3. **Honest naming without breaking keys.** Class names and config keys are
   unchanged. The package docstring
   (`strategies/__init__.py`) now states what each strategy *actually does*
   today, not the paper it is named after, and `PrefPOStrategy` is exported for
   parity. Renaming the classes/keys is explicitly rejected (see Alternatives).

## Consequences

- **Positive.** CoT guidance now reflects a proven trajectory instead of a
  generic checklist, so it can genuinely transfer a strong provider's approach.
  PrefPO stops diluting sections with off-topic rules — directly removing a
  documented source of degenerate growth. The docs no longer over-claim, which
  matters because the config surface implied four sophisticated optimizers when
  three are heuristics.
- **Neutral.** No protocol, config-key, schema, or reward-loop change; existing
  candidates and stored `strategy_name` rows keep working. Additive and
  independently revertible.
- **Cost.** The `SECTION_RELEVANT_FAILURES` map is a hand-maintained taxonomy;
  a new evolvable section that uses PrefPO should be added to it (or it runs
  unscoped by default).

## Alternatives Considered

- **Rename classes/keys to truthful names** (`KNNFewShotStrategy`,
  `ReasoningTransferStrategy`). Rejected: the keys are persisted in candidate
  rows and referenced across `victor/agent/`, so a rename is a migration for a
  cosmetic gain. Honesty is achievable in docstrings.
- **Delete the heuristic strategies, keep only GEPA.** Rejected: they are cheap,
  deterministic, and offline — useful as pre-GEPA layers and on installs
  without a mutator budget. The problem was fidelity and scoping, not
  existence.
- **Make PrefPO / CoT LLM-backed.** Deferred: that is a larger change that
  overlaps GEPA's role and belongs with the FEP-0025 experiment work, not this
  fidelity pass.

## Validation

- `tests/unit/framework/rl/test_cot_distillation_strategy.py` (new) pins
  faithfulness: real tool names in order, consecutive-tool collapse, reasoning
  carried through, recovery step only when the failure occurred, and the
  generic fallback for detail-less traces.
- `tests/unit/framework/rl/test_prefpo_strategy.py` gains section-scope cases:
  tool-discipline failures do **not** leak into `CONCISE_MODE_GUIDANCE`, while
  on-topic (`verbosity`) failures still drive it and `GROUNDING_RULES` still
  receives tool-discipline guidance.
