# RL Specs: Deep Impact Analysis & Thought Experiments

**Date:** 2026-11-15  
**Purpose:** Stress-test the five RL improvement specs against second-order effects, cascading failures, and foundational assumptions.

---

## Executive Summary

The conflation is **worse than initially documented** but **less catastrophic than it appears**:

- **4.6% of all outcomes** (8,913 of 191,688) come from conflated learners
- **95.4% of data is clean** — tool_selector (103K), prompt_template (62K), semantic_threshold (12K) record actual task outcomes
- The **asymmetric pattern** proves it's the CHECKS failing, not the tasks:
  - `grounding_threshold`: 0% on `analysis` (117) and `code_generation` (67), but 72.7% on `general` (1,777)
  - `continuation_patience`: 0% on `code_generation` (43), but 100% on `analysis` (137) and `debugging` (43)

If code_generation tasks were actually failing, ALL learners would show low success. The fact that only specific learner/task combinations show 0% proves the learners' internal checks are task-type-dependent.

---

## Spec 1: RL Measurement Integrity — Deep Analysis

### The Asymmetric Conflation Pattern (Smoking Gun)

```
grounding_threshold:
  analysis:        0% (117 outcomes)  ← grounding checks ALWAYS fail for analysis
  code_generation: 0% (67 outcomes)   ← grounding checks ALWAYS fail for code gen
  general:        72.7% (1,777 outcomes) ← grounding checks sometimes pass for general

continuation_patience:
  code_generation: 0% (43 outcomes)   ← patience checks ALWAYS fail for code gen
  analysis:      100% (137 outcomes)  ← patience checks ALWAYS pass for analysis
  debugging:     100% (43 outcomes)   ← patience checks ALWAYS pass for debugging
```

**Interpretation:** The grounding threshold is too strict for analysis/code_generation responses. The patience threshold is too strict for code_generation tasks. These are **real learning signals** — the learners are correctly identifying that their thresholds need adjustment for these task types. The problem is that `success=0` in `rl_outcome` makes it look like the TASK failed, when actually the CHECK failed.

### Thought Experiment 1: What if we "fix" the measurement but break the learning?

**Scenario:** We add `outcome_category='learner_internal'` and filter these out of task-level metrics.

**Risk:** The learners use `rl_outcome` data to learn. If we reclassify these outcomes, do the learners still get the signal they need?

**Analysis:**
- `grounding_threshold` builds context keys from `provider:response_type` (from metadata), NOT from `rl_outcome.task_type`
- `continuation_patience` builds context keys from `provider:model:task_type` — it DOES use task_type
- **Conclusion:** The learning itself is fine (learners use metadata). The OBSERVABILITY is broken (dashboards show wrong numbers).

**Impact of fix:**
- ✅ Dashboards show correct task-level success rates
- ✅ Provider comparisons are no longer skewed
- ⚠️ `continuation_patience` learner may need to switch to using metadata for task_type
- ✅ `grounding_threshold` learner is unaffected (already uses metadata)

### Thought Experiment 2: What if the conflation is actually HELPING?

**Scenario:** The 0% success for grounding checks on code_generation tells us the grounding threshold is too strict for code. If we "fix" the measurement, do we lose this signal?

**Analysis:**
- The learner already knows the threshold is too strict — it's updating Beta parameters based on fp/fn rates
- The `rl_outcome.success=0` is redundant — the learner already has the signal via `result_type='fp'` in metadata
- **Conclusion:** The conflation doesn't help learning. It only hurts observability.

### Thought Experiment 3: What if we retroactively classify historical data?

**Scenario:** `UPDATE rl_outcome SET outcome_category='learner_internal' WHERE learner_id IN ('grounding_threshold', 'continuation_patience', ...)`

**Risk:**
- Any existing dashboard or query that intentionally wants learner-internal outcomes breaks
- The `rl_outcome` table is the central fact table — changing semantics is dangerous
- **Mitigation:** Add the column with default='task', only set 'learner_internal' for NEW outcomes. Document that historical data is conflated.

### Foundational Needs

1. **The `rl_outcome` table serves two masters:** observability (dashboards) and learning (learner updates). These have different requirements:
   - Observability needs clean task-level metrics
   - Learning needs detailed per-check outcomes
   - **Current design conflates them.** The fix must separate these concerns without breaking either.

2. **Learners must not depend on `rl_outcome.task_type` for context keys.** If they do, changing task_type breaks learning. Audit all learners:
   - `grounding_threshold`: Uses metadata ✅
   - `continuation_patience`: Uses task_type ⚠️ — needs migration
   - `continuation_prompts`: Uses task_type ⚠️ — needs migration
   - `quality_weights`: Uses task_type ⚠️ — needs migration

3. **The 4.6% conflation fraction is small but poisonous.** Even though 95.4% of data is clean, the 4.6% contaminates any query that doesn't filter by learner_id. A single `SELECT task_type, AVG(success) FROM rl_outcome GROUP BY task_type` gives wrong answers.

### Second-Order Effects

1. **Provider routing decisions may be wrong.** If InferFlux's 81.6% success rate includes conflated grounding failures, the real rate might be higher. Conversely, if DeepSeek's 98.3% excludes conflated data, the comparison is unfair.

2. **Task complexity estimates may be skewed.** If code_generation shows 0% success, the system may allocate more budget than needed (or less, if it thinks the task is "impossible").

3. **User trust erodes.** If dashboards show "code_generation: 0% success," users lose confidence in the system, even though code generation actually works fine.

---

## Spec 2: Shell Tool Reliability — Deep Analysis

### Thought Experiment 1: What if 44% success is CORRECT?

**Scenario:** Shell commands include user-typed commands that may be intentionally wrong (`rm -rf /`, `git push --force`, typos).

**Analysis:**
- If agents are running legitimate commands, 44% is unacceptably low
- If agents are running exploratory/experimental commands, some failure is expected
- **Key question:** What fraction of shell calls are "legitimate" vs "exploratory"?

**Impact:**
- If most calls are legitimate → tool is broken, fix it
- If most calls are exploratory → 44% may be acceptable, but we need to know

**Foundational need:** We need to classify shell calls by intent before we can judge the success rate.

### Thought Experiment 2: What if the metadata bug is hiding the real problem?

**Scenario:** We fix metadata serialization and discover that 90% of failures are "command not found" or "permission denied."

**Analysis:**
- "Command not found" → agent is requesting tools that don't exist
- "Permission denied" → agent is running commands it shouldn't
- "Timeout" → commands are too slow, need better timeout handling
- **Impact:** The fix isn't in the shell tool — it's in the agent's command selection or the environment setup.

**Second-order effect:** Fixing metadata may reveal that the shell tool is fine but the agent is misbehaving. The real fix is in prompt engineering or tool availability.

### Thought Experiment 3: What if pre-validation blocks legitimate commands?

**Scenario:** We add command syntax validation that rejects commands with unmatched quotes.

**Risk:**
- Valid commands with complex quoting may be rejected
- Heredocs, nested quotes, and shell escaping are hard to validate
- **Impact:** Over-validation reduces tool utility, agents fall back to workarounds

**Mitigation:**
- Pre-validation should be conservative — only reject obviously broken commands
- Use allowlists for common patterns, not blocklists for edge cases
- Log rejected commands for analysis

### Foundational Needs

1. **Shell tool success rate is not a tool metric — it's a system metric.** It depends on:
   - Tool implementation (execution, error handling)
   - Agent behavior (command selection, error recovery)
   - Environment setup (available binaries, permissions, working directories)
   - **The spec must address all three layers, not just the tool.**

2. **Metadata serialization is a cross-cutting concern.** If `rl_tool_outcome.metadata` has serialization issues for shell, it probably affects other tools too. The fix should be in the recording infrastructure, not per-tool.

3. **Error classification is prerequisite for improvement.** Without knowing WHY commands fail, we can't fix them. The spec's Phase 2 (error classification) is actually Phase 1 — everything else depends on it.

---

## Spec 3: Graph Search Reliability — Deep Analysis

### Thought Experiment 1: What if graph search is fundamentally fragile?

**Scenario:** Embedding model mismatch is the dominant failure mode. Users switch models frequently, invalidating the index.

**Analysis:**
- If every model change requires re-indexing, graph search is unusable in practice
- Users won't re-index manually — they'll just stop using graph search
- **Impact:** Graph search may be a "demo feature" that works in controlled conditions but fails in production

**Second-order effect:** If graph search is fundamentally fragile, the fix isn't better thresholds — it's model-agnostic embeddings or automatic re-indexing.

### Thought Experiment 2: What if 36% success is actually GOOD?

**Scenario:** Semantic search naturally has lower success rates than text search. `code_search` (97%) is text matching; `graph_search` (36%) is semantic matching.

**Analysis:**
- Text search: query matches exact strings → high success
- Semantic search: query matches meaning → lower success (ambiguous queries, polysemy)
- **Key question:** What's the theoretical maximum for semantic search?

**Impact:**
- If 80% is the theoretical max, then 36% has room for improvement
- If 50% is the theoretical max, then 36% is close to optimal
- **Foundational need:** We need a baseline — what's the success rate of a "perfect" semantic search on this codebase?

### Thought Experiment 3: What if the graph is never indexed?

**Scenario:** Most users don't run `victor index` proactively. Graph search always fails for new users.

**Analysis:**
- If auto-indexing isn't implemented, graph search is dead on arrival for new users
- The spec's "no index" error message is a band-aid, not a fix
- **Impact:** Auto-indexing or lazy indexing is a PREREQUISITE, not an optimization

**Second-order effect:** If we implement auto-indexing, we need to handle:
- Large codebases (indexing takes minutes)
- Incremental updates (files change frequently)
- Multiple projects (each needs its own index)

### Foundational Needs

1. **Graph search depends on infrastructure that may not exist.** The spec assumes the graph is indexed. If it's not, nothing else matters. Auto-indexing is a foundational prerequisite.

2. **Embedding model stability is a hidden assumption.** If users switch models frequently, the index becomes stale. The spec must address model lifecycle management.

3. **Semantic search has inherent limitations.** The spec should establish a theoretical baseline and measure against it, not assume 100% is achievable.

---

## Spec 4: RL Learner Performance — Deep Analysis

### Thought Experiment 1: What if quality_weights 59% is the CEILING?

**Scenario:** Quality is inherently subjective. No linear weight scheme can perfectly predict human judgment.

**Analysis:**
- If quality scores are noisy (user feedback is sparse, completion scores are imperfect), then 59% may be the best a linear model can achieve
- **Key question:** Is the problem the model (linear weights) or the data (noisy quality scores)?

**Impact:**
- If the problem is the model → try non-linear models (neural nets, decision trees)
- If the problem is the data → fix the quality scoring first
- **Foundational need:** We need to measure the "signal-to-noise ratio" of quality scores before we can judge the learner.

### Thought Experiment 2: What if the learners are learning the WRONG thing?

**Scenario:** `quality_weights` learns to predict quality scores, but quality scores are flawed. `continuation_prompts` learns to select prompts, but "success" is poorly defined.

**Analysis:**
- Garbage in, garbage out — if the reward signal is wrong, the learner optimizes for the wrong thing
- **Key question:** Are quality scores correlated with actual user satisfaction?

**Impact:**
- If quality scores are poor proxies → fix the scoring before fixing the learner
- If "success" is poorly defined → clarify the definition before training

### Thought Experiment 3: What if exploration is the bottleneck?

**Scenario:** `continuation_prompts` has a limited prompt pool and doesn't explore enough.

**Analysis:**
- With only a few prompts, the learner can't adapt to different stuck patterns
- But adding more prompts increases the search space, requiring more exploration
- **Key tension:** Exploration vs. exploitation in a small search space

**Impact:**
- If the prompt pool is too small → expand it (but this is a band-aid)
- If exploration is insufficient → increase epsilon, use Thompson Sampling
- **Foundational need:** We need to understand the "prompt effectiveness distribution" — are there a few good prompts, or many mediocre ones?

### Foundational Needs

1. **Reward signal quality is prerequisite for learner improvement.** If quality scores are noisy, no learner can achieve high success. The spec must address reward signal quality before learner algorithms.

2. **Learner success rate may be bounded by task difficulty.** If some tasks are inherently hard, the learner can't succeed at all of them. The spec should establish per-task-type baselines.

3. **Exploration strategy must match the search space.** Epsilon-greedy works for small spaces; Thompson Sampling works for larger ones. The spec should match the strategy to the problem.

---

## Spec 5: Provider Routing Optimization — Deep Analysis

### Thought Experiment 1: What if InferFlux is slow because it's overloaded?

**Scenario:** InferFlux's 81.6% success rate is temporary — the server is under-resourced.

**Analysis:**
- Routing away from InferFlux doesn't fix the root cause (server overload)
- When InferFlux recovers, we'd route back, creating oscillation
- **Impact:** We're treating symptoms, not causes

**Second-order effect:** If we route away from InferFlux, it gets fewer requests, which might make it faster (less load), which might make us route back — creating a feedback loop.

### Thought Experiment 2: What if dynamic routing creates feedback loops?

**Scenario:** We route away from a provider → it gets fewer outcomes → less data → less confidence → we route back → oscillation.

**Analysis:**
- This is a classic control theory problem — the system needs damping
- **Mitigation:** Add cooldown periods, hysteresis, and minimum sample sizes before switching

**Impact:**
- Without damping, the system oscillates between providers
- With damping, the system converges to the best provider
- **Foundational need:** The routing system needs control theory principles (stability, convergence, damping).

### Thought Experiment 3: What if the "best" provider changes by task type?

**Scenario:** DeepSeek is best for code_generation but worst for analysis.

**Analysis:**
- Static routing (always use DeepSeek) is suboptimal
- Per-task-type routing is essential
- **Impact:** The spec's Phase 2 (task-aware routing) is not optional — it's foundational

**Second-order effect:** If different providers excel at different tasks, we need:
- Per-task-type performance tracking (already in the spec)
- Per-task-type routing decisions (already in the spec)
- Per-task-type failover (not in the spec — add it)

### Thought Experiment 4: What if cost-aware routing sacrifices quality?

**Scenario:** We route to a cheaper provider to save money, but quality drops.

**Analysis:**
- Cost-per-quality-point is the right metric, not cost-per-call
- But users may not care about quality-per-dollar — they may want absolute quality
- **Impact:** Cost-aware routing must have a quality floor (don't route below X quality)

**Foundational need:** The spec must distinguish between "cost-effective" and "cheap." Cost-effective maximizes quality-per-dollar; cheap minimizes cost regardless of quality.

### Foundational Needs

1. **Provider performance is non-stationary.** Providers change over time (updates, load, outages). The routing system must adapt continuously, not just once.

2. **Routing decisions have second-order effects on provider performance.** Routing away from a provider reduces its load, which may improve its performance, creating a feedback loop. The system needs damping.

3. **Task-aware routing is foundational, not optional.** Different providers excel at different tasks. Static routing is suboptimal.

4. **Cost-aware routing must have quality floors.** Don't sacrifice quality for cost below a threshold.

---

## Cross-Spec Interactions

### Interaction 1: Measurement Integrity → Provider Routing

If we fix the measurement conflation, provider performance metrics change. InferFlux's 81.6% might actually be higher (if some failures were conflated grounding checks). This affects routing decisions.

**Action:** Fix measurement integrity FIRST, then use corrected data for provider routing.

### Interaction 2: Shell Reliability → Graph Search Reliability

Both specs need metadata instrumentation. The shell spec's metadata infrastructure can be reused for graph search.

**Action:** Implement metadata infrastructure once, reuse across specs.

### Interaction 3: Learner Performance → Measurement Integrity

If we fix the measurement conflation, learner success rates change. `quality_weights` might actually be performing better than 59% (if some failures were conflated).

**Action:** Fix measurement integrity FIRST, then evaluate learner performance.

### Interaction 4: Provider Routing → All Other Specs

Provider routing depends on accurate performance metrics. If metrics are wrong (due to conflation), routing decisions are wrong.

**Action:** Fix measurement integrity FIRST, then implement provider routing.

---

## Recommended Implementation Order (Revised)

Based on the deep analysis, the order should be:

1. **P0: RL Measurement Integrity** (1 week)
   - Foundational prerequisite for all other specs
   - Without correct metrics, all other improvements are built on sand

2. **P1: Metadata Infrastructure** (1 week)
   - Shared by shell and graph search specs
   - Implement once, reuse across specs

3. **P1: Shell Tool Error Classification** (1 week)
   - Depends on metadata infrastructure
   - Prerequisite for shell reliability improvements

4. **P1: Graph Search Instrumentation** (1 week)
   - Depends on metadata infrastructure
   - Prerequisite for graph search improvements

5. **P2: RL Learner Performance** (2 weeks)
   - Depends on corrected metrics (from P0)
   - Can proceed in parallel with P1 work

6. **P2: Provider Routing** (2 weeks)
   - Depends on corrected metrics (from P0)
   - Should be last — it's the most complex and depends on all other fixes

---

## Open Questions for Team Discussion

1. **Is 4.6% conflation acceptable?** Or do we need to fix it before proceeding with other work?

2. **Should we retroactively classify historical data?** Or just fix forward?

3. **What's the theoretical maximum for semantic search?** We need a baseline before we can judge graph search performance.

4. **What's the theoretical maximum for quality prediction?** We need to know if 59% is good or bad.

5. **Should provider routing have a quality floor?** Or should it optimize purely for cost-effectiveness?

6. **How do we prevent routing oscillation?** What damping mechanisms should we use?

7. **Should we implement auto-indexing for graph search?** Or is manual indexing acceptable?

8. **What's the right success rate for shell commands?** Is 44% acceptable if most failures are user error?

---

## Conclusion

The deep analysis reveals that:

1. **The conflation is real but limited** — 4.6% of data, not catastrophic
2. **The asymmetry pattern proves the learners are working** — they're correctly identifying task-type-dependent threshold issues
3. **The observability is broken, not the learning** — learners use metadata, not `rl_outcome.task_type`
4. **Measurement integrity is the foundational prerequisite** — everything else depends on correct metrics
5. **Metadata infrastructure is a shared need** — implement once, reuse across specs
6. **Provider routing is the most complex** — it depends on all other fixes and has second-order effects

The specs are sound, but the implementation order matters. Fix measurement first, then metadata infrastructure, then the individual reliability improvements, and finally provider routing.

---

## Implementation ledger (2026-11-15)

Status rows for the revised implementation order above. Update at each material
milestone with exact PR/commit evidence; do not duplicate in other ledgers.

| ID | Increment | State | Evidence |
|----|-----------|-------|----------|
| RLM-1 | P0 RL measurement integrity (`outcome_category`, FEP-0046, metrics column fix) | ACTIVE — owner: vjsingh1984 session, branch `feat/rl-outcome-category`, baseline `origin/develop@e8c70a6e4` | FEP-0046 draft; tests for taxonomy partition, precedence chain, v10 migration, lazy guard, metrics regression |
| RLM-2 | P1 shared metadata infra + shell error classification (re-aimed at `tool_executor._emit_rl_tool_event`; `rl_tool_outcome` is legacy, zero writers) | TODO | blocked on RLM-1 merge |
| RLM-3 | P1 graph-search instrumentation + index health | TODO | blocked on RLM-1 merge |
| RLM-4 | P2 learner performance (quality_weights, continuation_prompts) | DATA-GATED | needs ≥2 weeks of post-RLM-1 categorized outcomes |
| RLM-5 | P2 provider routing (task-aware, damped) | DATA-GATED | last; needs corrected per-provider metrics |
