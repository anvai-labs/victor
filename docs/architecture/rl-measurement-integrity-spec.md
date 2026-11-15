# RL Measurement Integrity Spec

**Status:** Draft  
**Created:** 2026-11-15  
**Author:** Victor RL Team  
**Priority:** P0 (Critical)

---

## Problem Statement

Analysis of 191K RL outcomes in `~/.victor/victor.db` revealed a **measurement integrity bug** that invalidates task-level success metrics for certain task types.

### The Bug

The `code_generation` task type shows **0% success rate** across 110 outcomes. Investigation revealed this is NOT an actual code generation failure — it's a measurement conflation bug.

**Root Cause:** The `_dispatch_to_learner` method in `victor/core/rl/hooks.py` creates `RLOutcome` records with:
- `task_type` = the underlying task type (e.g., "code_generation")
- `success` = the learner's internal check result (e.g., grounding check pass/fail)

When `grounding_threshold` or `continuation_patience` learners record outcomes during a code_generation task, they record their *own* internal success (grounding check pass/fail, patience check pass/fail) but tag it with the underlying task's `task_type`. This conflates learner-internal outcomes with task outcomes.

### Evidence

```sql
-- All code_generation outcomes come from learner-internal checks
SELECT learner_id, COUNT(*) FROM rl_outcome 
WHERE task_type = 'code_generation' 
GROUP BY learner_id;

-- Result:
-- continuation_patience | 43
-- grounding_threshold   | 67
-- (Total: 110, all from learner-internal checks, NOT actual task outcomes)
```

### Impact

1. **Invalid metrics:** Task-level success rates are wrong for any task type where learner-internal checks are recorded
2. **Misleading analysis:** "code_generation 0% success" triggered investigation, but the real issue is measurement, not performance
3. **Learning corruption:** If learners use task_type to build context keys, they may be learning from conflated data
4. **Provider comparisons:** Provider performance metrics may be skewed if learner-internal outcomes are included

---

## Current Architecture

### Event Flow

```
1. Component emits RLEvent (e.g., GROUNDING_CHECK)
   - event.task_type = "code_generation" (underlying task)
   - event.success = False (grounding check failed)

2. RLHookRegistry._dispatch_to_learner() converts to RLOutcome
   - outcome.task_type = event.task_type  ← "code_generation"
   - outcome.success = event.success      ← False (grounding check)

3. Coordinator records to rl_outcome table
   - learner_id = "grounding_threshold"
   - task_type = "code_generation"
   - success = False

4. Analysis queries rl_outcome WHERE task_type = 'code_generation'
   - Sees 0% success → thinks code generation is broken
   - But these are grounding check failures, not task failures
```

### Affected Learners

| Learner | Event Type | What `success` Means | Conflation |
|---------|-----------|---------------------|------------|
| `grounding_threshold` | GROUNDING_CHECK | Grounding check passed | Yes |
| `continuation_patience` | CONTINUATION_ATTEMPT | Patience check passed | Yes |
| `continuation_prompts` | CONTINUATION_PROMPT | Prompt selection succeeded | Likely |
| `quality_weights` | QUALITY_ASSESSED | Quality assessment succeeded | Likely |
| `tool_selector` | TOOL_EXECUTED | Tool execution succeeded | **No** (correct) |
| `mode_transition` | MODE_TRANSITION | Mode transition succeeded | **No** (correct) |

Learners that check *internal system state* (grounding, patience, quality) conflate their check result with task success. Learners that track *external actions* (tool execution, mode transition) correctly record task outcomes.

---

## Proposed Solution

### Option A: Add `outcome_category` Field (Recommended)

Add a new field to `RLOutcome` and `rl_outcome` table to distinguish outcome types.

**Schema Change:**
```sql
ALTER TABLE rl_outcome ADD COLUMN outcome_category TEXT DEFAULT 'task';
-- Values: 'task' (actual task outcome), 'learner_internal' (learner check)
```

**Code Changes:**
1. Add `outcome_category: str = "task"` to `RLOutcome` dataclass
2. Update `_dispatch_to_learner` to set category based on event type:
   ```python
   LEARNER_INTERNAL_EVENTS = {
       RLEventType.GROUNDING_CHECK,
       RLEventType.CONTINUATION_ATTEMPT,
       RLEventType.CONTINUATION_PROMPT,
       RLEventType.QUALITY_ASSESSED,
   }
   
   outcome_category = (
       "learner_internal" if event.type in LEARNER_INTERNAL_EVENTS else "task"
   )
   ```
3. Update analysis queries to filter by `outcome_category = 'task'` for task-level metrics

**Pros:**
- Clean separation of concerns
- Backward compatible (default value)
- Enables precise filtering in analysis
- No breaking changes to existing learners

**Cons:**
- Requires schema migration
- Need to update existing analysis queries
- Historical data remains conflated (can't retroactively fix)

### Option B: Use Learner-Specific Task Types

Change `_dispatch_to_learner` to use learner-specific task types for internal outcomes.

**Code Changes:**
```python
EVENT_TASK_TYPE_MAP = {
    RLEventType.GROUNDING_CHECK: "learner:grounding_check",
    RLEventType.CONTINUATION_ATTEMPT: "learner:patience_check",
    RLEventType.CONTINUATION_PROMPT: "learner:prompt_selection",
    RLEventType.QUALITY_ASSESSED: "learner:quality_assessment",
}

task_type = EVENT_TASK_TYPE_MAP.get(event.type, event.task_type)
```

**Pros:**
- No schema change required
- Simple to implement
- Immediately separates outcomes in queries

**Cons:**
- Breaks learners that use task_type for context keys (e.g., `continuation_patience`)
- Lossy — can't easily query "all outcomes during code_generation tasks"
- Hacky — task_type field is overloaded

### Option C: Separate Tables

Create separate tables for learner-internal outcomes vs task outcomes.

**Schema Changes:**
```sql
CREATE TABLE rl_learner_outcome (
    -- Same schema as rl_outcome but for learner-internal checks
    learner_id TEXT,
    check_type TEXT,  -- 'grounding', 'patience', 'quality'
    ...
);
```

**Pros:**
- Clean separation
- No ambiguity

**Cons:**
- Major schema change
- Requires migrating existing data
- Breaks existing queries
- Overkill for the problem

---

## Recommended Approach: Option A

**Rationale:** Option A provides the cleanest separation with minimal disruption. The `outcome_category` field makes the distinction explicit and queryable, while remaining backward compatible.

### Implementation Plan

#### Phase 1: Schema Migration (1 day)

1. Add `outcome_category` column to `rl_outcome` table
   - File: `victor/core/schema.py`
   - Add migration step in `MigrationStep` list
   - Default value: `'task'` for backward compatibility

2. Add field to `RLOutcome` dataclass
   - File: `victor/framework/rl/base.py`
   - Add `outcome_category: str = "task"`

3. Update `RLCoordinator.record_outcome` to persist the field
   - File: `victor/core/rl/coordinator.py`
   - Include `outcome_category` in INSERT statement

#### Phase 2: Event Dispatch (1 day)

1. Define learner-internal event types
   - File: `victor/core/rl/hooks.py`
   - Add `LEARNER_INTERNAL_EVENTS` constant

2. Update `_dispatch_to_learner` to set category
   - File: `victor/core/rl/hooks.py`
   - Set `outcome.outcome_category` based on event type

3. Preserve original task_type in metadata
   - Add `original_task_type` to metadata for learners that need it
   - Example: `continuation_patience` uses task_type for context keys

#### Phase 3: Analysis Updates (1 day)

1. Update RL analysis queries to filter by `outcome_category`
   - File: `docs/analysis/rl-data-summary-2026-11-15.md`
   - Add note about historical data conflation
   - Provide corrected queries

2. Update learner implementations (if needed)
   - Verify learners don't rely on task_type for learner-internal outcomes
   - Update `continuation_patience` to use `original_task_type` from metadata

#### Phase 4: Testing (1 day)

1. Add unit tests for outcome category
   - File: `tests/unit/rl/test_outcome_measurement.py`
   - Test that learner-internal events set `outcome_category = 'learner_internal'`
   - Test that task events set `outcome_category = 'task'`

2. Add integration test for analysis queries
   - Verify task-level metrics exclude learner-internal outcomes
   - Verify learner-level metrics include all outcomes

3. Regression tests
   - Ensure existing learners still work correctly
   - Verify no breaking changes

### Verification

1. **Schema migration:**
   ```bash
   python -c "from victor.core.schema import Schema; print(Schema.RL_OUTCOME)"
   # Should include outcome_category column
   ```

2. **Outcome recording:**
   ```sql
   SELECT outcome_category, COUNT(*) FROM rl_outcome 
   GROUP BY outcome_category;
   -- Should see both 'task' and 'learner_internal' after fix
   ```

3. **Analysis queries:**
   ```sql
   -- Task-level metrics (excluding learner-internal)
   SELECT task_type, ROUND(100.0 * SUM(success) / COUNT(*), 1) as success_pct
   FROM rl_outcome
   WHERE outcome_category = 'task'
   GROUP BY task_type;
   
   -- Learner-level metrics (all outcomes)
   SELECT learner_id, ROUND(100.0 * SUM(success) / COUNT(*), 1) as success_pct
   FROM rl_outcome
   GROUP BY learner_id;
   ```

---

## Open Questions

1. **Historical data:** Should we attempt to retroactively classify existing outcomes?
   - Option: Use `learner_id` to infer category for historical data
   - Risk: May misclassify some outcomes
   - Recommendation: Leave historical data as-is, document the issue

2. **Learner context keys:** Should learners use `original_task_type` from metadata?
   - `continuation_patience` currently uses `task_type` for context keys
   - If we change `task_type` to "learner:patience_check", it loses per-task-type learning
   - Recommendation: Preserve original task_type in metadata, update learners to use it

3. **Default category:** Should new outcomes default to 'task' or require explicit category?
   - Current plan: Default to 'task' for backward compatibility
   - Alternative: Require explicit category (breaking change)
   - Recommendation: Default to 'task', add lint warning if not set

4. **Monitoring:** Should we add alerts for measurement anomalies?
   - Example: Alert if a task_type shows 0% success (likely measurement bug)
   - Recommendation: Add as follow-up work

---

## Acceptance Criteria

- [ ] `outcome_category` column added to `rl_outcome` table
- [ ] `RLOutcome` dataclass includes `outcome_category` field
- [ ] `_dispatch_to_learner` sets category based on event type
- [ ] Unit tests verify correct category assignment
- [ ] Analysis queries updated to filter by category
- [ ] Documentation updated with corrected metrics
- [ ] No breaking changes to existing learners
- [ ] Historical data documented as conflated

---

## Follow-Up Work

1. **Retroactive classification:** Attempt to classify historical outcomes using `learner_id`
2. **Measurement monitoring:** Add alerts for measurement anomalies (0% success, etc.)
3. **Learner-specific specs:** Create specs for improving underperforming learners (quality_weights, continuation_prompts)
4. **Provider routing:** Use corrected metrics to optimize provider routing

---

## References

- RL Data Analysis: `docs/analysis/rl-data-summary-2026-11-15.md`
- FEP-0033: RL Subsystem Relocation
- FEP-0017: Prompt Optimization Reward Loop
- Schema: `victor/core/schema.py`
- Hooks: `victor/core/rl/hooks.py`
- Base: `victor/framework/rl/base.py`
