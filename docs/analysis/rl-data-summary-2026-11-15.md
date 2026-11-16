# Victor RL Data Analysis Summary

**Date:** 2026-11-15  
**Database:** `~/.victor/victor.db` (163MB)  
**Data Period:** 2026-04-24 to 2026-11-10 (~7 months)

---

## Executive Summary

Victor has accumulated **191,688 RL outcomes** and **176,479 state transitions** across 13 learner types. The data reveals strong performance patterns (95%+ success for most learners), actionable tool preferences, and mode transition strategies that can strengthen the codebase.

---

## 1. Data Volume Overview

| Table | Rows | Description |
|-------|------|-------------|
| `rl_outcome` | 191,688 | Task outcomes with success/quality metrics |
| `rl_transition` | 176,479 | State→action→state transitions |
| `rl_tool_outcome` | 12,035 | Per-tool success/failure tracking |
| `rl_task_stat` | 3,585 | Aggregated task statistics |
| `rl_grounding_history` | 3,837 | Grounding threshold decisions |
| `rl_mode_history` | 1,652 | Mode transition events |
| `rl_semantic_stat` | 411 | Semantic search effectiveness |
| `rl_provider_stat` | 181 | Provider latency/success tracking |
| `rl_tool_q` | 41 | Learned tool Q-values |
| `rl_mode_q` | 51 | Learned mode transition Q-values |
| `rl_grounding_param` | 7 | Learned grounding parameters per provider |

---

## 2. Learner Performance Summary

| Learner | Total | Success % | Avg Quality | Notes |
|---------|-------|-----------|-------------|-------|
| `tool_selector` | 103,262 | **93.8%** | 0.892 | High volume, strong performance |
| `prompt_template` | 62,320 | **100%** | 0.500 | Perfect success, baseline quality |
| `semantic_threshold` | 11,559 | **99.9%** | 0.776 | Near-perfect threshold selection |
| `quality_weights` | 4,631 | 59.1% | 0.640 | ⚠️ Needs improvement |
| `grounding_threshold` | 3,927 | **80.0%** | 0.879 | Good quality when successful |
| `workflow_execution` | 3,231 | **98.8%** | 0.982 | Excellent workflow completion |
| `mode_transition` | 1,342 | **100%** | 0.500 | Perfect transitions |
| `prompt_optimizer` | 455 | **81.8%** | 0.774 | Solid optimization |
| `model_selector` | 348 | **100%** | 0.803 | Perfect model selection |
| `continuation_patience` | 229 | **81.2%** | 0.505 | Decent patience calibration |
| `team_composition` | 212 | **96.2%** | 0.867 | Strong team decisions |
| `continuation_prompts` | 126 | 69.8% | 0.499 | ⚠️ Needs improvement |
| `user_feedback` | 46 | **100%** | 0.700 | Limited but perfect |

### Key Insights:
- **`quality_weights` (59.1% success)** and **`continuation_prompts` (69.8% success)** are underperforming
- **`workflow_execution` (98.8% success, 0.982 quality)** is the star performer
- Most learners exceed 80% success rate

---

## 3. Provider/Model Performance (Recent: Sep-Nov 2026)

| Provider | Model | Calls | Success % | Avg Quality |
|----------|-------|-------|-----------|-------------|
| zai | glm-5.2 | 15,985 | **96.3%** | 0.828 |
| zai | glm-5.1 | 11,776 | **96.5%** | 0.839 |
| deepseek | deepseek-chat | 6,839 | **98.3%** | 0.856 |
| openai | qwen3.7-plus | 6,713 | **94.1%** | 0.826 |
| deepseek | deepseek-v4-flash | 4,799 | **93.1%** | 0.819 |
| moonshot | kimi-k3 | 1,929 | **97.6%** | 0.833 |
| zai | glm-5.3 | 1,560 | **96.1%** | 0.771 |
| inferflux | qwen3-coder-30b | 1,502 | 81.6% | 0.657 |
| anthropic | claude-sonnet-4-6 | 668 | **95.2%** | 0.846 |
| xai | grok-4-1-fast | 708 | **96.8%** | 0.769 |

### Key Insights:
- **DeepSeek Chat** has highest success rate (98.3%) with good quality
- **Z.ai GLM models** dominate volume with consistent 96%+ success
- **InferFlux qwen3-coder-30b** underperforms (81.6% success, 0.657 quality)
- **Claude Sonnet 4.6** performs well (95.2% success, 0.846 quality)

---

## 4. Tool Effectiveness (Learned Q-Values)

| Tool | Q-Value | Selections | Success % | Interpretation |
|------|---------|------------|-----------|----------------|
| `timed_tool` | 0.950 | 158 | 100% | 🟢 Highly reliable |
| `ls` | 0.939 | 1,801 | 98.3% | 🟢 Core navigation |
| `docs_coverage` | 0.933 | 88 | 100% | 🟢 Perfect for docs |
| `code_search` | 0.931 | 1,086 | 97.1% | 🟢 Essential search |
| `write_code` | 0.924 | 84 | 100% | 🟢 Code generation |
| `symbol` | 0.923 | 69 | 95.7% | 🟢 Symbol operations |
| `project_overview` | 0.918 | 131 | 96.9% | 🟢 Context gathering |
| `read` | 0.918 | 6,189 | 94.4% | 🟢 Most used, reliable |
| `graph_query` | 0.915 | 56 | 100% | 🟢 Graph operations |
| `graph` | 0.899 | 793 | 87.6% | 🟡 Good but lower success |
| `edit` | 0.704 | 77 | 67.5% | 🟠 Moderate reliability |
| `test` | 0.664 | 13 | 92.3% | 🟡 Limited data |
| `shell` | N/A | 271 | **44.3%** | 🔴 Low success rate |
| `graph_search` | N/A | 66 | **36.4%** | 🔴 Low success rate |
| `read_file` | N/A | 144 | **50.0%** | 🔴 Low success rate |

### Key Insights:
- **`read`, `ls`, `code_search`** are the workhorses (high volume, high success)
- **`shell` (44.3% success)** and **`graph_search` (36.4% success)** need attention
- **`edit` (67.5% success)** has room for improvement despite high Q-value

---

## 5. Mode Transition Patterns

### Top Transitions (explore mode):

| From | Action | To | Count | Success % | Avg Reward |
|------|--------|----|-------|-----------|------------|
| explore | explore:+0 | explore | 946 | 100% | -0.165 |
| explore | plan:+0 | plan | 290 | **88.3%** | 0.329 |
| explore | build:+0 | build | 223 | **89.7%** | 0.190 |
| explore | complete:+0 | complete | 193 | **85.0%** | 0.317 |

### Mode Q-Values (Learned Preferences):

| State | Action | Q-Value | Visits | Interpretation |
|-------|--------|---------|--------|----------------|
| explore:create_simple:low | explore:+0 | 0.347 | 48 | 🟢 Stay in explore |
| explore:search:low | explore:+0 | 0.341 | 9 | 🟢 Continue exploring |
| explore:design:low | complete:+0 | 0.129 | 3 | 🟡 Consider completing |
| explore:edit:low | plan:+0 | 0.116 | 69 | 🟡 Transition to plan |
| explore:analyze:low | build:+0 | -0.125 | 120 | 🔴 Avoid building |
| explore:analyze:low | complete:+0 | -0.133 | 136 | 🔴 Avoid completing |

### Key Insights:
- **`explore → plan`** (88.3% success) and **`explore → build`** (89.7% success) are strong transitions
- **`explore → complete`** (85.0% success) is viable but slightly lower
- Analysis tasks should **stay in explore** longer before transitioning
- High visit counts with negative Q-values indicate **learned avoidance patterns**

---

## 6. Grounding Threshold Learning

### Learned Parameters by Provider:

| Provider | Threshold | Alpha | Beta | Samples |
|----------|-----------|-------|------|---------|
| unknown:general | 0.7 | 128.5 | 5.7 | 1,359 |
| zai:general | 0.7 | 111.0 | 1.0 | 1,108 |
| anthropic:general | 0.7 | 81.3 | 1.0 | 803 |
| deepseek:general | 0.7 | 29.0 | 1.0 | 281 |
| xai:general | 0.7 | 8.2 | 1.0 | 72 |
| unknown:code_generation | 0.8 | 6.9 | 1.4 | 63 |
| ollama:general | 0.7 | 14.9 | 1.0 | 151 |

### Grounding Results:

| Result | Count | Avg Threshold | Avg Reward |
|--------|-------|---------------|------------|
| TN (True Negative) | 3,762 | 0.701 | 0.1 |
| TP (True Positive) | 23 | 0.722 | 0.1 |
| FP (False Positive) | 46 | 0.709 | -1.0 |
| FN (False Negative) | 6 | 0.700 | -2.0 |

### Key Insights:
- **0.7 threshold** is well-calibrated across providers
- **Code generation** uses higher threshold (0.8) — more conservative
- **False negatives are costly** (-2.0 reward) — system is appropriately conservative
- **High alpha values** (128.5, 111.0) indicate strong prior confidence

---

## 7. Task Type Performance

| Task Type | Total | Success % | Avg Quality |
|-----------|-------|-----------|-------------|
| general | 176,764 | **95.0%** | 0.745 |
| default | 11,534 | **99.9%** | 0.777 |
| analyze | 1,274 | **93.5%** | 0.978 |
| edit | 325 | **94.8%** | 0.991 |
| design | 103 | **97.1%** | 0.985 |
| create | 120 | **95.8%** | 0.977 |
| search | 76 | **97.4%** | 0.978 |
| analysis | 358 | 62.3% | 0.496 |
| code_generation | 110 | **0.0%** | 0.470 |

### Key Insights:
- **`code_generation` has 0% success** — critical issue to investigate
- **`analysis` (62.3% success)** needs improvement
- **`edit`, `design`, `create`** have excellent quality scores (0.97+)
- **`general` dominates volume** (176K outcomes)

---

## 8. Monthly Trends

| Month | Outcomes | Success % | Avg Quality |
|-------|----------|-----------|-------------|
| 2026-11 | 16,871 | **93.0%** | 0.869 |
| 2026-10 | 2,559 | **94.8%** | 0.872 |
| 2026-09 | 4,879 | 83.6% | 0.719 |
| 2026-08 | 1,221 | **93.0%** | 0.854 |
| 2026-07 | 50,115 | **96.6%** | 0.893 |
| 2026-06 | 20,353 | **95.5%** | 0.789 |
| 2026-05 | 62,083 | **95.7%** | 0.652 |
| 2026-04 | 33,607 | **94.6%** | 0.617 |

### Key Insights:
- **Quality is improving** over time (0.617 → 0.869)
- **September 2026 dip** (83.6% success) needs investigation
- **November 2026** shows strong performance (93% success, 0.869 quality)

---

## 9. Actionable Recommendations

### 🔴 Critical (Fix Immediately):

1. **`code_generation` task type has 0% success**
   - Investigate root cause in task routing or execution
   - Check if this is a data collection issue or actual failures
   - Priority: P0

2. **`shell` tool has 44.3% success rate**
   - Review shell command execution logic
   - Add better error handling and validation
   - Consider sandboxing or command whitelisting
   - Priority: P1

3. **`graph_search` has 36.4% success rate**
   - Investigate search query construction
   - Check embedding model compatibility
   - Review result filtering logic
   - Priority: P1

### 🟡 High Priority (Improve Soon):

4. **`quality_weights` learner at 59.1% success**
   - Review weight adjustment algorithm
   - Check if rewards are properly calibrated
   - Consider adjusting learning rate
   - Priority: P2

5. **`continuation_prompts` at 69.8% success**
   - Analyze prompt selection strategy
   - Review context window management
   - Priority: P2

6. **`edit` tool at 67.5% success**
   - Improve edit diff generation
   - Add better conflict detection
   - Priority: P2

7. **InferFlux `qwen3-coder-30b` underperforms (81.6% success)**
   - Compare with other providers on same tasks
   - Check if model is properly configured
   - Consider adjusting routing preferences
   - Priority: P2

### 🟢 Strengthen (Leverage Success):

8. **`workflow_execution` is excellent (98.8% success, 0.982 quality)**
   - Document best practices
   - Apply patterns to other learners
   - Priority: P3

9. **`explore → plan` transition is strong (88.3% success)**
   - Encourage this transition in mode selection logic
   - Priority: P3

10. **Quality improvement trend (0.617 → 0.869)**
    - Continue current optimization strategies
    - Document what changed between May and November
    - Priority: P3

---

## 10. Data Quality Notes

- **User feedback is sparse**: Only 46 outcomes with user feedback, 95 auto-evaluated
- **No corrections recorded**: `correction` field is empty across all outcomes
- **Mock/test data present**: `mock_provider` and `test` entries should be excluded from production analysis
- **Grounding data is conservative**: High TN count (3,762) vs TP (23) suggests system avoids false positives

---

## 11. Next Steps

1. **Immediate**: Investigate `code_generation` 0% success rate
2. **This week**: Fix `shell` and `graph_search` tool reliability
3. **This month**: Improve `quality_weights` and `continuation_prompts` learners
4. **Ongoing**: Continue monitoring quality trends and provider performance

---

**Generated by:** Qwen Code  
**Analysis Date:** 2026-11-15  
**Data Source:** `~/.victor/victor.db`

---

## Post-script (2026-11-15, FEP-0046): corrected metrics

The numbers above are the **conflated** view. FEP-0046 (PR #1848) added
`outcome_category` to `rl_outcome` and backfilled history deterministically;
re-running the key queries on the migrated database corrects them:

| Metric | Conflated (above) | Corrected |
|---|---|---|
| Overall success | 95.13% | **93.92%** (task rows only) |
| Learner-internal share | invisible | **44.3%** (85,032 / 191,688 rows) |
| `code_generation` success | 0% (110 rows) | **no task rows** — all 110 were grounding/patience internal checks |
| `session_summary` | 100% (95 rows) | **no task rows** — aggregation artifacts |
| Genuine task-level rows | 191,688 | **106,656** (`general` 106,610 + `feedback` 46) |

Two structural findings the corrected data exposes:

1. **Fine-grained task types (`analysis`, `edit`, `design`, ...) existed only in
   learner-internal rows** — task-type granularity was an artifact of the
   conflation. Real task-typed outcomes must come from the tool-execution
   emit path (RLM-2's structured channel).
2. **`prompt_template`'s 100% success / 62K outcomes were `prompt_used`
   bookkeeping events with hardcoded `success=True`** — not task outcomes.

Corrected query shape for any future analysis of this table:

```sql
SELECT task_type, ROUND(100.0*SUM(success)/COUNT(*),1) AS success_pct, COUNT(*)
FROM rl_outcome
WHERE outcome_category = 'task'   -- exclude learner-internal checks
GROUP BY task_type;
```

Per-learner analyses intentionally do NOT filter by category — a learner's
internal check pass-rate is its own legitimate metric.
