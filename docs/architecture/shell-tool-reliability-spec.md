# Shell Tool Reliability Spec

**Status:** Draft  
**Created:** 2026-11-15  
**Author:** Victor RL Team  
**Priority:** P1 (High)

---

## Problem Statement

The `shell` tool has a **44.3% success rate** (271 total calls: 120 successes, 151 failures) based on RL outcome data from `~/.victor/victor.db`. This is significantly lower than other core tools:

| Tool | Success Rate | Total Calls |
|------|-------------|-------------|
| `read` | 94.4% | 6,251 |
| `ls` | 98.3% | 1,814 |
| `code_search` | 97.1% | 1,098 |
| `shell` | **44.3%** | 271 |

### Evidence

```sql
-- Shell tool outcomes by success
SELECT task_type, success, COUNT(*) as count,
       ROUND(AVG(quality_score), 3) as avg_quality,
       ROUND(AVG(reward), 3) as avg_reward
FROM rl_tool_outcome
WHERE tool_name = 'shell'
GROUP BY task_type, success;

-- Result:
-- general | 0 | 151 | 0.133 | 0.077  (failures)
-- general | 1 | 120 | 0.941 | 0.938  (successes)
```

### Impact

1. **User experience:** Shell commands fail nearly half the time, breaking workflows
2. **Agent reliability:** Agents can't reliably execute CLI commands (git, docker, make, etc.)
3. **RL learning:** Low success rate poisons tool_selector Q-values, reducing shell tool selection
4. **Productivity:** Users must retry commands or fall back to manual execution

---

## Root Cause Analysis

### Metadata Serialization Issue

The `rl_tool_outcome.metadata` field shows serialization problems:

```sql
-- Shell failure metadata patterns
SELECT json_extract(metadata, '$.error_type') as error_type,
       json_extract(metadata, '$.exit_code') as exit_code,
       COUNT(*) as count
FROM rl_tool_outcome
WHERE tool_name = 'shell' AND success = 0
GROUP BY error_type, exit_code;

-- Result:
-- (empty) | (empty) | 80   ← No error info recorded
-- (empty) | str       | 71   ← Type serialization issue
```

**Finding:** The metadata is not being properly serialized. Error information is lost, making debugging impossible.

### Potential Root Causes

1. **Metadata serialization bug:** Tool outcome recording doesn't properly JSON-serialize metadata
2. **Error classification missing:** Shell tool doesn't categorize failures (timeout, permission, not found, etc.)
3. **Command validation gaps:** Invalid commands reach execution instead of being caught early
4. **Timeout handling:** Commands may be timing out without proper error reporting
5. **Working directory issues:** `cwd` validation may be failing silently

### What We Don't Know

Due to poor metadata, we cannot determine:
- What types of commands fail (git, docker, make, etc.)
- Why they fail (timeout, permission, syntax, missing binary)
- Which working directories are problematic
- Whether failures are user errors or system bugs

---

## Proposed Solution

### Phase 1: Fix Metadata Recording (P0 — 2 days)

**Goal:** Ensure shell tool outcomes include complete, serializable metadata.

#### 1.1 Define Shell Outcome Metadata Schema

```python
@dataclass
class ShellOutcomeMetadata:
    """Structured metadata for shell tool outcomes."""
    # Command info
    cmd: str                          # The command that was executed
    cwd: str                          # Working directory
    action: str                       # "read", "write", "exec", "network"
    
    # Execution result
    exit_code: int                    # Process exit code
    timed_out: bool                   # Whether command timed out
    duration_ms: float                # Execution time
    
    # Error classification
    error_category: str               # "none", "timeout", "permission", "not_found", 
                                      # "syntax", "exit_code", "blocked", "validation"
    error_message: str                # Human-readable error
    stderr_snippet: str               # First 500 chars of stderr
    
    # Safety/policy
    blocked_by_policy: bool           # Whether ShellSafetyPolicy blocked it
    readonly_violation: bool          # Whether it violated readonly mode
    dangerous_command: bool           # Whether it was flagged as dangerous
    
    # Output summary
    stdout_lines: int                 # Number of stdout lines
    stderr_lines: int                 # Number of stderr lines
    stdout_truncated: bool            # Whether stdout was truncated
    stderr_truncated: bool            # Whether stderr was truncated
```

#### 1.2 Update Shell Tool to Record Structured Metadata

**File:** `victor/tools/bash.py`

```python
async def shell(cmd, cwd, timeout, ...) -> Dict[str, Any]:
    start_time = time.monotonic()
    
    try:
        result = await _execute_shell(cmd, cwd, timeout, ...)
        duration_ms = (time.monotonic() - start_time) * 1000
        
        metadata = ShellOutcomeMetadata(
            cmd=cmd,
            cwd=cwd,
            action=action,
            exit_code=result.get("return_code", -1),
            timed_out=result.get("timed_out", False),
            duration_ms=duration_ms,
            error_category=_classify_error(result),
            error_message=result.get("error", ""),
            stderr_snippet=result.get("stderr", "")[:500],
            blocked_by_policy=False,
            readonly_violation=False,
            dangerous_command=False,
            stdout_lines=len(result.get("stdout", "").splitlines()),
            stderr_lines=len(result.get("stderr", "").splitlines()),
            stdout_truncated=result.get("stdout_truncated", False),
            stderr_truncated=result.get("stderr_truncated", False),
        )
        
        # Record outcome with structured metadata
        _record_tool_outcome(
            tool_name="shell",
            success=result.get("return_code", -1) == 0,
            metadata=asdict(metadata),  # Proper serialization
        )
        
        return result
        
    except Exception as e:
        # Record failure with error classification
        metadata = ShellOutcomeMetadata(
            cmd=cmd, cwd=cwd, action=action,
            exit_code=-1, timed_out=False,
            duration_ms=(time.monotonic() - start_time) * 1000,
            error_category=_classify_exception(e),
            error_message=str(e),
            ...
        )
        _record_tool_outcome(
            tool_name="shell",
            success=False,
            metadata=asdict(metadata),
        )
        raise
```

#### 1.3 Fix Metadata Serialization in Tool Outcome Recording

**File:** `victor/framework/rl/hooks.py` or wherever `record_tool_outcome` is called

Ensure metadata is JSON-serialized before storage:

```python
def _record_tool_outcome(tool_name: str, success: bool, metadata: dict, ...):
    from victor.core.json_utils import json_dumps
    
    # Ensure metadata is properly serialized
    serialized_metadata = json_dumps(metadata)
    
    # Store in database
    db.execute(
        "INSERT INTO rl_tool_outcome (tool_name, success, metadata, ...) VALUES (?, ?, ?, ...)",
        (tool_name, success, serialized_metadata, ...),
    )
```

### Phase 2: Error Classification (P1 — 2 days)

**Goal:** Categorize shell failures to identify patterns.

#### 2.1 Error Category Classifier

```python
def _classify_error(result: dict) -> str:
    """Classify shell command error into category."""
    if result.get("timed_out"):
        return "timeout"
    
    exit_code = result.get("return_code", -1)
    stderr = result.get("stderr", "")
    
    if exit_code == 0:
        return "none"
    
    # Permission errors
    if "permission denied" in stderr.lower() or exit_code == 126:
        return "permission"
    
    # Command not found
    if "command not found" in stderr.lower() or exit_code == 127:
        return "not_found"
    
    # Syntax errors
    if "syntax error" in stderr.lower():
        return "syntax"
    
    # Non-zero exit code (generic failure)
    if exit_code > 0:
        return "exit_code"
    
    return "unknown"

def _classify_exception(e: Exception) -> str:
    """Classify exception into error category."""
    error_str = str(e).lower()
    
    if "timeout" in error_str:
        return "timeout"
    if "permission" in error_str:
        return "permission"
    if "not found" in error_str or "no such file" in error_str:
        return "not_found"
    if "blocked" in error_str or "policy" in error_str:
        return "blocked"
    if "validation" in error_str or "invalid" in error_str:
        return "validation"
    
    return "unknown"
```

#### 2.2 Add Error Category to Outcome Recording

Update shell tool to call classifier and include category in metadata.

### Phase 3: Command Pre-Validation (P2 — 3 days)

**Goal:** Catch invalid commands before execution.

#### 3.1 Command Syntax Validation

```python
def _validate_command_syntax(cmd: str) -> tuple[bool, str]:
    """Validate command syntax before execution.
    
    Returns:
        (is_valid, error_message)
    """
    # Check for empty command
    if not cmd or not cmd.strip():
        return False, "Empty command"
    
    # Check for placeholder commands
    if _is_placeholder_cmd(cmd):
        return False, f"'{cmd}' is a placeholder, not a real command"
    
    # Check for unmatched quotes
    if _has_unmatched_quotes(cmd):
        return False, "Unmatched quotes in command"
    
    # Check for invalid shell operators
    if _has_invalid_operators(cmd):
        return False, "Invalid shell operator"
    
    return True, ""
```

#### 3.2 Working Directory Validation

```python
def _validate_cwd(cwd: str) -> tuple[bool, str]:
    """Validate working directory exists and is accessible."""
    import os
    
    if not os.path.exists(cwd):
        return False, f"Working directory does not exist: {cwd}"
    
    if not os.path.isdir(cwd):
        return False, f"Not a directory: {cwd}"
    
    if not os.access(cwd, os.R_OK | os.X_OK):
        return False, f"Working directory not accessible: {cwd}"
    
    return True, ""
```

#### 3.3 Early Validation in Shell Tool

```python
async def shell(cmd, cwd, ...):
    # Pre-validation
    valid, error = _validate_command_syntax(cmd)
    if not valid:
        _record_tool_outcome(
            tool_name="shell",
            success=False,
            metadata={"error_category": "validation", "error_message": error, ...},
        )
        return {"success": False, "error": error, ...}
    
    valid, error = _validate_cwd(cwd)
    if not valid:
        _record_tool_outcome(
            tool_name="shell",
            success=False,
            metadata={"error_category": "validation", "error_message": error, ...},
        )
        return {"success": False, "error": error, ...}
    
    # Proceed with execution
    ...
```

### Phase 4: Timeout Handling Improvements (P2 — 2 days)

**Goal:** Better timeout detection and reporting.

#### 4.1 Explicit Timeout Tracking

```python
async def _execute_shell(cmd, cwd, timeout, ...):
    import asyncio
    
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )
            return {
                "return_code": proc.returncode,
                "stdout": stdout.decode(),
                "stderr": stderr.decode(),
                "timed_out": False,
            }
        except asyncio.TimeoutError:
            proc.kill()
            return {
                "return_code": -1,
                "stdout": "",
                "stderr": f"Command timed out after {timeout}s",
                "timed_out": True,
            }
    except Exception as e:
        return {
            "return_code": -1,
            "stdout": "",
            "stderr": str(e),
            "timed_out": False,
        }
```

### Phase 5: Analysis and Monitoring (P3 — 1 day)

**Goal:** Track shell tool reliability over time.

#### 5.1 Success Rate Dashboard

Add to `victor doctor` or monitoring dashboard:

```python
def check_shell_reliability():
    """Check shell tool success rate from RL data."""
    db = get_victor_db()
    
    recent = db.execute("""
        SELECT success, COUNT(*) as count
        FROM rl_tool_outcome
        WHERE tool_name = 'shell'
          AND timestamp > datetime('now', '-7 days')
        GROUP BY success
    """).fetchall()
    
    total = sum(row["count"] for row in recent)
    successes = next((row["count"] for row in recent if row["success"]), 0)
    
    if total == 0:
        return {"status": "no_data", "message": "No shell outcomes in last 7 days"}
    
    success_rate = successes / total
    
    if success_rate < 0.6:
        return {
            "status": "critical",
            "message": f"Shell success rate is {success_rate:.1%} (target: >80%)",
            "success_rate": success_rate,
        }
    elif success_rate < 0.8:
        return {
            "status": "warning",
            "message": f"Shell success rate is {success_rate:.1%} (target: >80%)",
            "success_rate": success_rate,
        }
    else:
        return {
            "status": "healthy",
            "message": f"Shell success rate is {success_rate:.1%}",
            "success_rate": success_rate,
        }
```

#### 5.2 Error Category Breakdown

```python
def get_shell_error_breakdown():
    """Get breakdown of shell errors by category."""
    db = get_victor_db()
    
    return db.execute("""
        SELECT json_extract(metadata, '$.error_category') as category,
               COUNT(*) as count
        FROM rl_tool_outcome
        WHERE tool_name = 'shell'
          AND success = 0
          AND timestamp > datetime('now', '-30 days')
        GROUP BY category
        ORDER BY count DESC
    """).fetchall()
```

---

## Implementation Plan

### Sprint 1: Metadata Fix (Week 1)

- [ ] Define `ShellOutcomeMetadata` dataclass
- [ ] Update shell tool to record structured metadata
- [ ] Fix metadata serialization in `_record_tool_outcome`
- [ ] Add unit tests for metadata recording
- [ ] Verify metadata round-trips correctly in database

### Sprint 2: Error Classification (Week 2)

- [ ] Implement `_classify_error` function
- [ ] Implement `_classify_exception` function
- [ ] Integrate classifier into shell tool
- [ ] Add unit tests for error classification
- [ ] Test with real shell commands

### Sprint 3: Pre-Validation (Week 3)

- [ ] Implement command syntax validation
- [ ] Implement working directory validation
- [ ] Add early validation to shell tool
- [ ] Add unit tests for validation
- [ ] Test with edge cases

### Sprint 4: Timeout Handling (Week 4)

- [ ] Implement explicit timeout tracking
- [ ] Update shell tool to use new timeout handling
- [ ] Add unit tests for timeout scenarios
- [ ] Test with long-running commands

### Sprint 5: Monitoring (Week 5)

- [ ] Add shell reliability check to `victor doctor`
- [ ] Add error category breakdown query
- [ ] Document monitoring approach
- [ ] Set up alerts for critical success rates

---

## Verification

### Metadata Verification

```bash
# Query recent shell outcomes
sqlite3 ~/.victor/victor.db "
SELECT 
    json_extract(metadata, '$.error_category') as category,
    json_extract(metadata, '$.exit_code') as exit_code,
    json_extract(metadata, '$.timed_out') as timed_out,
    COUNT(*) as count
FROM rl_tool_outcome
WHERE tool_name = 'shell'
  AND timestamp > datetime('now', '-1 day')
GROUP BY category, exit_code, timed_out;
"
```

### Success Rate Verification

```bash
# Check shell success rate
sqlite3 ~/.victor/victor.db "
SELECT 
    ROUND(100.0 * SUM(success) / COUNT(*), 1) as success_pct,
    COUNT(*) as total
FROM rl_tool_outcome
WHERE tool_name = 'shell'
  AND timestamp > datetime('now', '-7 days');
"
```

### Error Category Verification

```bash
# Check error breakdown
sqlite3 ~/.victor/victor.db "
SELECT 
    json_extract(metadata, '$.error_category') as category,
    COUNT(*) as count
FROM rl_tool_outcome
WHERE tool_name = 'shell'
  AND success = 0
  AND timestamp > datetime('now', '-7 days')
GROUP BY category
ORDER BY count DESC;
"
```

---

## Acceptance Criteria

- [ ] Shell tool outcomes include complete, structured metadata
- [ ] Metadata is properly JSON-serialized (no "str" type markers)
- [ ] Error categories are recorded for all failures
- [ ] Command pre-validation catches syntax errors before execution
- [ ] Working directory validation catches invalid paths
- [ ] Timeout handling properly tracks and reports timeouts
- [ ] Shell success rate improves from 44% to >80%
- [ ] `victor doctor` reports shell reliability status
- [ ] Unit tests cover all new functionality
- [ ] No regression in existing shell tool functionality

---

## Open Questions

1. **Timeout defaults:** What should the default timeout be? Current is None (unlimited).
   - Option: 30s for read actions, 60s for write/exec actions
   - Recommendation: Start with 60s default, make configurable per-profile

2. **Command allowlists:** Should we maintain allowlists for common commands?
   - Option: Whitelist git, docker, make, npm, cargo, etc.
   - Risk: Maintenance burden, may block legitimate commands
   - Recommendation: Don't use allowlists; rely on safety policy + validation

3. **Output truncation:** How much stdout/stderr should we capture?
   - Current: Unlimited (can cause memory issues)
   - Option: Cap at 10,000 lines for stdout, 2,000 for stderr
   - Recommendation: Add configurable limits, default to current behavior

4. **Retry logic:** Should we automatically retry failed commands?
   - Option: Retry once on timeout, never on other errors
   - Risk: May mask real issues, waste time
   - Recommendation: Don't auto-retry; let agent decide

---

## Follow-Up Work

1. **Shell command patterns:** Analyze which commands fail most often
2. **Provider-specific issues:** Check if certain providers have higher shell failure rates
3. **Working directory patterns:** Identify problematic working directories
4. **Timeout optimization:** Tune timeout values based on actual command durations

---

## References

- RL Data Analysis: `docs/analysis/rl-data-summary-2026-11-15.md`
- Shell Tool: `victor/tools/bash.py`
- FEP-0013: Shell Safety Policy
- Tool Outcome Table: `rl_tool_outcome`
