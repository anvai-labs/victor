# Task Completion Detection

## Overview

Victor uses **explicit signal-based task completion detection** to determine when tasks are finished, replacing the legacy buffer/size-based heuristics that were prone to false positives and unnecessary continuation loops.

## How It Works

### Completion Signals

When Victor completes a task, it uses explicit markers to signal completion:

- **`_DONE_`** - File operations (creation, modification, deletion)
  - `_DONE_ Created test.py`
  - `_DONE_ Modified src/api.py`

- **`_TASK_DONE_`** - Bug fixes and task completion
  - `_TASK_DONE_ Fixed authentication bug`
  - `_TASK_DONE_ Resolved circular import`

- **`_SUMMARY_`** - Analysis and research tasks
  - `_SUMMARY_ Key findings: 1. Performance bottleneck in database queries...`
  - `_SUMMARY_ Analysis complete. Three issues identified...`

- **`_BLOCKED_`** - Cannot complete task
  - `_BLOCKED_ Missing API key for service X`

### Response Phases

Victor distinguishes between different phases of work:

1. **EXPLORATION** - Reading files, searching codebase
2. **SYNTHESIS** - Summarizing, planning, preparing output
3. **FINAL_OUTPUT** - Delivering answer, completed work
4. **BLOCKED** - Cannot complete, needs user input

### Confidence Levels

Completion detection uses graded confidence levels:

- **HIGH** - Active signal detected (_DONE_, _TASK_DONE_, _SUMMARY_) → **Immediate completion**
- **MEDIUM** - File modifications + passive signal → **Continue with logging**
- **LOW** - Only passive phrase → **Continue**
- **NONE** - No signals → **Use normal logic**

## Configuration

### Enable/Disable Signal-Based Completion

Edit your Victor profile or `.env` file:

```bash
# Enable signal-based completion (default, recommended)
VICTOR_USE_SIGNAL_BASED_COMPLETION=true

# Disable to use legacy buffer/size heuristics (not recommended)
VICTOR_USE_SIGNAL_BASED_COMPLETION=false
```

Or in code:

```python
from victor.config import Settings

settings = Settings()
settings.use_signal_based_completion = True  # Enable (default)
settings.use_signal_based_completion = False  # Disable (legacy)
```

## Task Type Completion Signals

Each task type has specific completion signal instructions:

### Coding Tasks
- **create**: `_DONE_ Created <filename>`
- **edit**: `_DONE_ Modified <filename>`
- **bug_fix**: `_TASK_DONE_ Fixed <issue>`
- **refactor**: `_DONE_ Refactored <component>`

### Analysis Tasks
- **analyze**: `_SUMMARY_ Analysis complete. Key findings: <summary>`
- **codebase_review**: `_SUMMARY_ Review complete. Issues found: <list>`

### Research Tasks
- **research**: `_SUMMARY_ Research complete. Findings: <summary>`
- **literature_review**: `_SUMMARY_ Literature review complete. <key_points>`

### DevOps Tasks
- **deploy**: `_DONE_ Deployed <service> to <environment>`
- **dockerfile**: `_DONE_ Created Dockerfile for <service>`

## Examples

### Example 1: File Creation

**User Request**: "Create a Python hello world script"

**Victor's Response**:
```
I'll create a hello world script for you.

[Creates file]

_DONE_ Created hello.py with a simple hello world program.
```

**Result**: Task completes immediately after `_DONE_` signal.

### Example 2: Bug Fix

**User Request**: "Fix the authentication bug in login.py"

**Victor's Response**:
```
Let me examine the login.py file first...

[Reading and analyzing]

I found the issue. The token validation is missing.

[Edits the file]

_TASK_DONE_ Fixed authentication bug in login.py
```

**Result**: Task completes after `_TASK_DONE_` signal.

### Example 3: Analysis Task

**User Request**: "Analyze the performance bottlenecks in the codebase"

**Victor's Response**:
```
I'll analyze the codebase for performance issues...

[Exploring multiple files]

After analyzing the codebase, I've identified the following bottlenecks:

_SUMMARY_ Performance Analysis Complete:
1. Database queries: N+1 query problem in user_service.py
2. Caching: Missing cache headers for static assets
3. Algorithm: O(n²) complexity in search function

Recommendations:
- Implement query batching
- Add CDN caching
- Use binary search algorithm
```

**Result**: Task completes after `_SUMMARY_` signal.

## Benefits

### For Users

✅ **Clear Completion Markers** - See when tasks are done with explicit markers
✅ **Faster Completion** - No unnecessary continuation loops
✅ **Better Output** - Valid output not consumed as "thinking content"
✅ **Predictable Behavior** - Deterministic completion based on signals

### For Developers

✅ **No False Positives** - Explicit signals instead of heuristics
✅ **Debuggable** - Clear markers in logs showing completion reason
✅ **Testable** - Can test completion detection independently
✅ **Observable** - Structured events for monitoring

## Troubleshooting

### Task Not Completing

If a task doesn't complete when expected:

1. **Check for completion signal** - Ensure response contains `_DONE_`, `_TASK_DONE_`, or `_SUMMARY_`
2. **Enable debug logging** - Set `VICTOR_LOG_LEVEL=DEBUG` to see completion detection
3. **Verify feature flag** - Ensure `use_signal_based_completion=true`

### Task Completing Too Early

If a task completes prematurely:

1. **Check signal placement** - Ensure `_DONE_` is AFTER actual completion
2. **Review task type hints** - Some tasks may require specific output formats
3. **Disable signal-based completion** - Set `use_signal_based_completion=false` as fallback

### Force Legacy Mode

If you need to disable signal-based completion:

```python
from victor.config import Settings

settings = Settings()
settings.use_signal_based_completion = False
```

Or via environment variable:

```bash
export VICTOR_USE_SIGNAL_BASED_COMPLETION=false
victor chat
```

## Monitoring

### Completion Events

Victor emits structured events for completion detection:

```python
{
    "topic": "state.continuation.task_complete",
    "data": {
        "reason": "task_completion_detector_high_confidence",
        "confidence": "HIGH",
        "source": "TaskCompletionDetector"
    }
}
```

### Log Messages

Look for these log messages to understand completion decisions:

```
INFO: Task completion: HIGH confidence detected (active signal), forcing completion after this response
INFO: Task completion: MEDIUM confidence detected (file mods + passive signal)
INFO: ContinuationStrategy: HIGH confidence from TaskCompletionDetector - finishing
```

## Migration from Legacy Behavior

### What Changed

**Before (Legacy)**:
- Completion detected based on response size (500+ characters)
- "Natural completion" based on accumulated content length
- False positives when model provided long explanations
- No clear indication of task completion

**After (Signal-Based)**:
- Completion detected via explicit markers (_DONE_, _TASK_DONE_, _SUMMARY_)
- Response phase detection distinguishes thinking from output
- Deterministic completion when active signals present
- Clear markers show task is complete

### Rollback Plan

If you encounter issues:

1. **Immediate rollback**: Set `use_signal_based_completion=false`
2. **Report issue**: Include log output showing completion decision
3. **Gradual rollout**: Enable for specific task types first

## Best Practices

### For Users

✅ **Let Victor complete naturally** - Don't rush the completion signal
✅ **Review completion markers** - Check that `_DONE_` appears after actual work is done
✅ **Use appropriate signal type**:
   - File operations → `_DONE_`
   - Bug fixes → `_TASK_DONE_`
   - Analysis → `_SUMMARY_`

### For Developers

✅ **Add completion signals to prompts** - Include signal instructions in system prompts
✅ **Test completion detection** - Verify detector recognizes signals
✅ **Monitor completion confidence** - Check confidence levels in logs
✅ **Handle all confidence levels** - Support HIGH, MEDIUM, LOW, NONE appropriately

## Technical Details

### Architecture

This section describes the response-heuristic components. The
[unified agentic-loop diagram](../architecture.md#agenticloop) is the canonical execution
sequence: the shared EVALUATE and DECIDE phases own the run’s continuation decision.
A detector confidence score alone should not be read as a separate execution authority.

### Components

- **TaskCompletionDetector** (`victor/agent/task_completion.py`)
  - Detects completion signals in LLM responses
  - Calculates confidence levels
  - Determines response phases

- **ContinuationStrategy** (`victor/agent/continuation_strategy.py`)
  - Uses detector confidence for continuation decisions
  - Priority: Detector > Intent Classification > Legacy

- **IntentClassifier** (`victor/storage/embeddings/intent_classifier.py`)
  - Used as fallback for ambiguous cases
  - Only consulted when detector confidence is not HIGH

## References

- **Plan Document**: `/Users/vijaysingh/.claude/plans/joyful-gathering-badger.md`
- **Source Code**: `victor/agent/task_completion.py`
- **Protocol**: `victor/agent/protocols/task_completion.py`
- **Tests**: `tests/unit/agent/test_task_completion.py`
