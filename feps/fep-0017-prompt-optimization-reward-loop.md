---
fep: "0017"
title: "Close the prompt-optimization reward loop — serve, attribute, and reward evolved prompt candidates"
type: Standards Track
status: Implemented
created: 2026-07-14
modified: 2026-07-14
authors:
  - name: Vijaykumar Singh
    email: vijay@anvaiops.com
    github: vjsingh1984
reviewers: []
discussion: https://github.com/vjsingh1984/victor/discussions/0017
---

# FEP-0017: Close the prompt-optimization reward loop

## Summary

GEPA prompt evolution can *create* candidate prompts, and PR #519 fixed the
persist gate so `evolve()` now *stores* them. But the loop still does not turn:
stored candidates are never **served**, never **rewarded**, and never
**improve** — their Thompson posteriors stay at the prior
`alpha=beta=1.0, sample_count=0, is_active=0` forever. This FEP closes the loop
with three additive, mostly-local changes: (1) **serve** candidates via a
relaxed gate (proven OR benchmark-approved OR epsilon-explored); (2) **attribute**
each served turn's outcome through a new `PROMPT_CANDIDATE_USED` RL event wired
into the existing dispatch; (3) **reward** by fixing the `record_outcome`
chicken-and-egg so a `sample_count=0` candidate can be found by its served hash.

## Motivation

Observed in production (`/prompt-optimize` after #519): candidates are stored,
but `/prompt-optimize --status` shows every candidate at `Samples=0, Live=no`.
Three independent defects:

- **Serve gate** (`optimization_injector.py`, `init_synthesizer.py`): the gate
  `rec.confidence > 0.6 and not rec.is_baseline` is unsatisfiable for any
  candidate that has never been served, because
  `confidence = evidence_count / (MIN_SAMPLES_FOR_CONFIDENCE * 2)` and
  `evidence_count` is `sample_count` until rewards exist. Even benchmark-
  *approved* candidates stay blocked (their *live* `sample_count` is 0, so
  `is_baseline=True`).
- **`record_outcome` chicken-and-egg** (`prompt_optimizer.py`): it finds a
  candidate only if `is_active` or `sample_count > 0`. A fresh candidate has
  neither, so even if an outcome arrived it could never be attributed.
- **No emission/wiring**: `prompt_optimizer` is absent from `EVENT_TO_LEARNER`
  (`hooks.py`), and no code emits a prompt-candidate outcome event, so
  `record_outcome` is never called in production.

## Proposed Change

### Serve decision (`should_serve_candidate`)

A shared helper (`victor/framework/rl/learners/prompt_optimizer.py`) decides
whether to serve a recommendation:

```
serve = proven OR benchmark_approved OR (exploration_enabled AND rand() < epsilon)
  proven             = confidence > 0.6 AND NOT is_baseline   (unchanged exploit path)
  benchmark_approved = rec.metadata["benchmark_passed"]        (unblocks suite-validated)
  explore            = epsilon-greedy bootstrap of fresh candidates
```

`exploration_enabled` (default `True`) and `exploration_epsilon` (default `0.1`)
live on `PromptOptimizationSettings`. On serve, the injector calls
`learner.record_served(section, provider, hash)` so the candidate is rewardable
even before it has samples.

### Attribution (`PROMPT_CANDIDATE_USED`)

- New `RLEventType.PROMPT_CANDIDATE_USED`, mapped to `["prompt_optimizer"]` in
  `EVENT_TO_LEARNER` and to `EventType.PROGRESS` in the taxonomy.
- `RuntimeIntelligenceService.get_prompt_optimization_bundle` already computes
  per-turn served identities (`PromptOptimizationIdentity`); it now caches the
  rewardable ones (those carrying a `prompt_candidate_hash`).
- New `record_prompt_candidate_outcome(completion_score, success, ...)`: emits
  one `PROMPT_CANDIDATE_USED` event per cached identity, carrying
  `prompt_section`, `prompt_candidate_hash`, and `session_id`. Non-blocking,
  mirroring `response_quality._emit_quality_assessed_event`.
- The agentic loop calls it **once per `run()`** at turn-outcome finalization,
  guarded `try/except`, mirroring the existing single-fire
  `record_topology_outcome` call.

The existing `_dispatch_to_learner` already merges `event.metadata` into
`RLOutcome.metadata` and calls `coordinator.record_outcome("prompt_optimizer",
outcome)` — no new dispatch code is required.

### Reward (`record_outcome` resolution)

`record_outcome` resolves the candidate by hash first — from
`outcome.metadata["prompt_candidate_hash"]`, then the `_last_served` ledger —
before falling back to active / `sample_count > 0`. This unblocks a brand-new
candidate. The posterior update (`candidate.update(success)` → α/β/sample_count,
plus the EMA `completion_score`) is unchanged.

**Success is derived from the completion score, not the agentic-loop COMPLETE
flag.** `emit_prompt_candidate_outcome` computes
`success = reward_success_from_score(completion_score)` (`>= 0.5`). A mid-task
`CONTINUE` turn is not a prompt failure: scoring `COMPLETE`-only would make every
non-terminal turn penalize (`beta++`) the served candidate, biasing the posterior
toward *turn position* rather than prompt quality. Using the continuous score
removes that bias. (Deeper limitation — per-turn progress is still a confounded
proxy for prompt quality; session-level attribution remains the principled
longer-term signal, see Unresolved Questions.)

### Lifecycle (closed loop)

```
evolve() stores candidate (gen N)
  └─ chat turn: injector serves it (proven/approved/explore) → record_served()
       └─ turn completes → record_prompt_candidate_outcome(score, success)
            └─ PROMPT_CANDIDATE_USED → dispatch → record_outcome
                 └─ candidate α/β/sample_count update; confidence rises
                      └─ next turns serve it more when good (Thompson Sampling)
```

## Benefits

- The prompt-optimization loop actually closes: `/prompt-optimize` output
  enters live rotation, accumulates real outcome evidence, and good candidates
  win out over time via Thompson Sampling.
- Benchmark-approved candidates finally get served (the suite-validation path
  was dead due to `is_baseline`).
- Reuses the established, tested RL dispatch infra (`tool_selector`,
  `quality_weights`) — no parallel attribution mechanism.
- Additive and reversible (`exploration_enabled=false` restores prior behavior).

## Drawbacks and Alternatives

- **Exploration serves unvalidated prompts ε of the time.** Bounded upstream by
  the #519 structural gate (no corrupt candidates) and the low default ε=0.1;
  worst case a mediocre prompt that the reward signal then corrects.
- **Alternative considered — offline trace-attribution**: persist served
  identities with the session and attribute at `evolve()` time from collected
  traces, avoiding any hot-path event. Rejected for this FEP because it delays
  feedback to evolve cadence and touches ConversationStore/trace-collection;
  the event-based path is idiomatic and unit-testable. The offline model remains
  a viable fallback if the hot-path emit proves costly.
- **Alternative considered — drop the confidence gate entirely** (pure Thompson
  Sampling). Rejected: too aggressive for live serving; epsilon-exploration keeps
  a proven/exploit majority while bootstrapping.

## Unresolved Questions

- ~~**Streaming path**~~ — **resolved.** Both `run()` and `run_streaming()` now
  call the shared `AgenticLoop._emit_prompt_reward_outcome`, so buffered and
  streaming chat update the learner identically. This mattered more than the
  FEP anticipated: interactive chat is predominantly streaming, so had parity
  not landed the loop would have been closed almost nowhere in practice.
- **Multi-section credit assignment**: when several sections are served in one
  turn, each currently receives the same turn-level completion score.
  Per-section/counterfactual attribution is deferred.
- **Reward cadence vs. noise**: per-turn completion scores are noisy; whether a
  per-session aggregate would learn faster is left to measurement.

## Implementation Plan

Single PR (`fix/prompt-reward-loop`):

1. Learner (`prompt_optimizer.py`): `_last_served` ledger, `record_served`,
   `record_outcome` hash/last-served resolution, `should_serve_candidate`
   helper, `benchmark_passed` on recommendation metadata.
2. Wiring (`hooks.py`): `PROMPT_CANDIDATE_USED` event type +
   `EVENT_TO_LEARNER` + taxonomy.
3. Settings (`prompt_optimization_settings.py`): `exploration_enabled`,
   `exploration_epsilon`.
4. Serve gate (`optimization_injector.py`, `init_synthesizer.py`): use
   `should_serve_candidate`; call `record_served` on serve.
5. Emission (`runtime_intelligence.py`): cache served identities +
   `record_prompt_candidate_outcome`.
6. Agentic loop (`agentic_loop.py`): one guarded single-fire call at turn
   outcome.
7. Tests + this FEP.

## Migration Path

Additive and non-breaking. Defaults (`exploration_enabled=True`,
`exploration_epsilon=0.1`) opt users into the closed loop immediately; set
`exploration_enabled=false` to restore the pre-FEP serve-only-when-proven
behavior. No schema change — the existing `agent_prompt_candidate` columns
(alpha/beta/sample_count/completion_score) are reused. No data migration; the
candidates stored by #519 begin accumulating rewards on their next served turn.

## Compatibility

- No public API change. New event type and setting are additive.
- The persist gate (#519) remains the structural safety floor for explored
  candidates.
- Unchanged: GEPA/PrefPO/MIPROv2 strategies, Pareto frontiers, benchmark
  gating, `record_benchmark_result`, `promote_candidate`.
- The agentic-loop touch is one non-blocking guarded call; streaming path is
  unaffected (and not yet covered — see Unresolved Questions).

## References

- PR #519 — reconcile GEPA persist gate to structural-only (prerequisite).
- `victor/framework/rl/hooks.py` — `EVENT_TO_LEARNER`, `RLEvent`, dispatch.
- `victor/framework/rl/learners/prompt_optimizer.py` — `PromptOptimizerLearner`.
- `victor/agent/services/runtime_intelligence.py` —
  `PromptOptimizationIdentity`, `get_prompt_optimization_bundle`.
- GEPA methodology (ProTeGi / prompt evolution via execution traces).
