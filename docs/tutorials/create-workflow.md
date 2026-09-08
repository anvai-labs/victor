# Creating Workflows in Victor

Learn how to build powerful multi-agent workflows using Victor's YAML-first workflow system. This tutorial walks you through creating a complete code review workflow with conditions, parallel execution, and human-in-the-loop approvals.

## What You Will Build

By the end of this tutorial, you will have created a **code review workflow** that:

- Analyzes code for issues using an AI agent
- Runs security and style checks in parallel
- Routes tasks based on code quality conditions
- Requires human approval before merging
- Generates a final review report

## Prerequisites

- Victor installed (`pip install victor-ai`)
- Basic understanding of YAML syntax
- Familiarity with Victor's agent system

**Time estimate: 45 minutes**

---

## 1. Workflow Basics

### YAML Structure Overview

Victor workflows are defined in YAML files with a consistent structure:

```yaml
workflows:
  workflow_name:
    description: "What this workflow does"
    metadata:
      key: value
    nodes:
      - id: node_id
        type: agent|compute|condition|parallel|transform|hitl
        # node-specific configuration
        next:
          - next_node_id
```

### Node Types Summary

| Type | Purpose | Uses LLM |
|------|---------|----------|
| `agent` | Execute tasks using AI agents with tools | Yes |
| `compute` | Execute tools directly without LLM reasoning | No |
| `condition` | Branch workflow based on conditions | No |
| `parallel` | Execute multiple nodes simultaneously | Varies |
| `transform` | Transform data between nodes | No |
| `hitl` | Pause for human approval or input | No |
| `team` | Spawn multi-agent teams | Yes |

### How Workflows Execute

1. **Start**: Execution begins at the first node (or specified `start_node`)
2. **Sequential**: Nodes execute in order based on `next` references
3. **Branching**: Condition nodes route to different paths based on context
4. **Parallel**: Parallel nodes execute children simultaneously, then merge results
5. **Human Gates**: HITL nodes pause execution until human responds
6. **Completion**: Workflow ends when reaching a node with no `next` or an `END` type

---

## 2. Step-by-Step: Simple Workflow

Let's start with a basic two-node workflow.

### Step 1: Create the YAML File

Create a file `workflows/simple_review.yaml`:

```yaml
workflows:
  simple_review:
    description: "Basic code review workflow"
    nodes:
      - id: analyze
        type: agent
        role: code_reviewer
        goal: "Analyze the provided code for issues and best practice violations"
        tool_budget: 15
        allowed_tools:
          - read
          - grep
          - code_search
        output_key: analysis_result
        next:
          - report

      - id: report
        type: agent
        role: reporter
        goal: "Generate a comprehensive review report based on the analysis"
        tool_budget: 5
        allowed_tools:
          - read
        output_key: review_report
```

### Step 2: Add Metadata

Metadata provides context for the workflow system:

```yaml
workflows:
  simple_review:
    description: "Basic code review workflow"
    metadata:
      category: code_quality
      version: "1.0"
      author: "team"
      tags:
        - review
        - quality
    nodes:
      # ... nodes from above
```

### Step 3: Configure Node Details

Each agent node supports these key properties:

```yaml
- id: analyze                    # Unique identifier
  type: agent                    # Node type
  name: "Code Analyzer"          # Human-readable name (optional)
  role: code_reviewer            # Agent role: researcher, executor, reviewer, writer, etc.
  goal: "Analyze code..."        # Task description (supports $ctx.variable substitution)
  tool_budget: 15                # Maximum tool calls allowed
  allowed_tools:                 # Specific tools to enable
    - read
    - grep
  input_mapping:                 # Map context keys to agent inputs
    file_path: target_file
  output_key: analysis_result    # Store output under this key
  timeout_seconds: 120           # Optional execution timeout
  llm_config:                    # Optional LLM overrides
    temperature: 0.3
  next:
    - report
```

### Step 4: Run the Workflow

Execute your workflow using the CLI:

```bash
# Validate the workflow
victor workflow validate workflows/simple_review.yaml

# Run the workflow
victor workflow run workflows/simple_review.yaml --workflow simple_review \
  --input '{"target_file": "src/main.py"}'
```

Or programmatically in Python:

```python
from victor.workflows.unified_compiler import compile_and_execute

result = await compile_and_execute(
    "workflows/simple_review.yaml",
    initial_state={"target_file": "src/main.py"},
    workflow_name="simple_review"
)

print(result.state.get("review_report"))
```

---

## 3. Adding Conditions

Conditions allow workflows to branch based on runtime state.

### Condition Nodes

```yaml
workflows:
  conditional_review:
    description: "Review with quality-based routing"
    nodes:
      - id: analyze
        type: agent
        role: code_reviewer
        goal: "Analyze code quality and count issues"
        tool_budget: 15
        output_key: analysis
        next:
          - check_quality

      - id: check_quality
        type: condition
        condition: "code_quality_check"    # References escape hatch function
        branches:
          excellent: approve_merge
          good: approve_merge
          acceptable: minor_fixes
          needs_improvement: major_fixes

      - id: minor_fixes
        type: agent
        role: fixer
        goal: "Apply minor code improvements"
        tool_budget: 10
        next:
          - final_report

      - id: major_fixes
        type: agent
        role: fixer
        goal: "Address major issues found in analysis"
        tool_budget: 20
        next:
          - re_analyze

      - id: re_analyze
        type: agent
        role: code_reviewer
        goal: "Re-analyze code after fixes"
        tool_budget: 10
        next:
          - check_quality

      - id: approve_merge
        type: agent
        role: approver
        goal: "Approve code for merging"
        tool_budget: 3
        next:
          - final_report

      - id: final_report
        type: agent
        role: reporter
        goal: "Generate final review summary"
        tool_budget: 5
```

### Escape Hatch Functions

Conditions reference Python functions defined in `escape_hatches.py`. Create one in your vertical or workflow directory:

```python
# escape_hatches.py
"""Escape hatches for review workflows."""

from typing import Any, Dict


def code_quality_check(ctx: Dict[str, Any]) -> str:
    """Assess code quality based on analysis results.

    Args:
        ctx: Workflow context containing:
            - analysis (dict): Analysis results from the analyzer

    Returns:
        "excellent", "good", "acceptable", or "needs_improvement"
    """
    analysis = ctx.get("analysis", {})

    # Extract metrics from analysis
    errors = analysis.get("errors", 0)
    warnings = analysis.get("warnings", 0)
    issues = analysis.get("issues", [])

    if errors == 0 and warnings == 0:
        return "excellent"

    if errors == 0 and warnings <= 3:
        return "good"

    if errors <= 2:
        return "acceptable"

    return "needs_improvement"


def tests_passed(ctx: Dict[str, Any]) -> str:
    """Check if tests are passing.

    Args:
        ctx: Workflow context with test_results

    Returns:
        "true" or "false"
    """
    test_results = ctx.get("test_results", {})
    failed = test_results.get("failed", 0)
    return "true" if failed == 0 else "false"


# Registry for the YAML loader
CONDITIONS = {
    "code_quality_check": code_quality_check,
    "tests_passed": tests_passed,
}

TRANSFORMS = {}
```

### Using Escape Hatches

When compiling, provide the condition registry:

```python
from victor.workflows.unified_compiler import UnifiedWorkflowCompiler
from escape_hatches import CONDITIONS, TRANSFORMS

compiler = UnifiedWorkflowCompiler()
graph = compiler.compile_yaml(
    "workflows/conditional_review.yaml",
    workflow_name="conditional_review",
    condition_registry=CONDITIONS,
    transform_registry=TRANSFORMS,
)

result = await graph.invoke({"target_file": "src/main.py"})
```

---

## 4. Parallel Execution

Run multiple analyses simultaneously for faster workflows.

### Parallel Nodes

```yaml
workflows:
  parallel_review:
    description: "Code review with parallel checks"
    nodes:
      - id: parallel_checks
        type: parallel
        parallel_nodes:
          - security_scan
          - style_check
          - complexity_analysis
        join_strategy: all        # Wait for all to complete
        next:
          - merge_results

      - id: security_scan
        type: agent
        role: security_analyst
        goal: "Scan code for security vulnerabilities"
        tool_budget: 15
        allowed_tools:
          - read
          - grep
          - code_search
        output_key: security_results

      - id: style_check
        type: agent
        role: style_reviewer
        goal: "Check code style and formatting"
        tool_budget: 10
        allowed_tools:
          - read
          - shell
        output_key: style_results

      - id: complexity_analysis
        type: agent
        role: analyst
        goal: "Analyze code complexity metrics"
        tool_budget: 10
        allowed_tools:
          - read
          - code_search
        output_key: complexity_results

      - id: merge_results
        type: transform
        transform: "merge_review_results"
        next:
          - final_report

      - id: final_report
        type: agent
        role: reporter
        goal: "Compile all findings into a comprehensive report"
        tool_budget: 5
```

### Join Strategies

| Strategy | Behavior |
|----------|----------|
| `all` | Wait for all parallel nodes to complete |
| `any` | Continue when any one node completes |
| `majority` | Continue when majority complete (>50%) |
| `merge` | Merge all results into combined output |

### Transform for Merging Results

Add to your `escape_hatches.py`:

```python
def merge_review_results(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Merge results from parallel review checks.

    Args:
        ctx: Context with parallel results

    Returns:
        Merged results dictionary
    """
    security = ctx.get("security_results", {})
    style = ctx.get("style_results", {})
    complexity = ctx.get("complexity_results", {})

    # Combine all issues
    all_issues = []
    all_issues.extend(security.get("issues", []))
    all_issues.extend(style.get("issues", []))
    all_issues.extend(complexity.get("issues", []))

    return {
        "merged_results": {
            "total_issues": len(all_issues),
            "issues": all_issues,
            "security_score": security.get("score", 100),
            "style_score": style.get("score", 100),
            "complexity_score": complexity.get("score", 50),
        }
    }


TRANSFORMS = {
    "merge_review_results": merge_review_results,
}
```

---

## 5. Human-in-the-Loop

Add human approval gates for critical workflow decisions.

### HITL Approval Nodes

```yaml
workflows:
  reviewed_merge:
    description: "Code review with human approval"
    nodes:
      - id: analyze
        type: agent
        role: code_reviewer
        goal: "Analyze code changes thoroughly"
        tool_budget: 20
        output_key: analysis
        next:
          - human_review

      - id: human_review
        type: hitl
        hitl_type: approval
        prompt: "Review the following code analysis. Approve to proceed with merge."
        context_keys:
          - analysis
          - files_changed
        timeout: 600              # 10 minutes
        fallback: abort           # abort, continue, skip, retry
        next:
          - merge_code

      - id: merge_code
        type: agent
        role: executor
        goal: "Merge the approved changes"
        tool_budget: 5
```

### HITL Node Types

Victor supports several HITL interaction types:

```yaml
# Binary approval
- id: approve_changes
  type: hitl
  hitl_type: approval
  prompt: "Approve these changes?"
  timeout: 300
  fallback: abort

# Multiple choice
- id: select_action
  type: hitl
  hitl_type: choice
  prompt: "How should we proceed?"
  choices:
    - "Deploy to staging"
    - "Deploy to production"
    - "Cancel deployment"
  timeout: 300
  fallback: continue
  default_value: "Deploy to staging"

# Free-form input
- id: get_feedback
  type: hitl
  hitl_type: input
  prompt: "Enter additional review comments:"
  timeout: 600
  fallback: skip

# Review with modification
- id: review_plan
  type: hitl
  hitl_type: review
  prompt: "Review and optionally modify the implementation plan:"
  context_keys:
    - implementation_plan
    - files_to_modify
  timeout: 900
  fallback: abort

# Simple confirmation
- id: confirm_proceed
  type: hitl
  hitl_type: confirmation
  prompt: "Press Enter to continue..."
  timeout: 60
  fallback: continue
```

### Fallback Behaviors

| Fallback | Behavior on Timeout |
|----------|---------------------|
| `abort` | Stop workflow execution |
| `continue` | Continue with default value |
| `skip` | Skip this node entirely |
| `retry` | Retry the request |

---

## 6. Escape Hatches

When workflow logic exceeds what YAML can express, use Python escape hatches.

### When to Use Python

Use escape hatches for:

- Complex conditional logic with multiple factors
- Data transformations requiring Python libraries
- Integration with external services
- Business logic that changes frequently

### Creating escape_hatches.py

For any vertical, create `escape_hatches.py` in the vertical directory:

```python
# victor/myvertical/escape_hatches.py
"""Escape hatches for MyVertical YAML workflows."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


# =============================================================================
# Condition Functions
# =============================================================================

def should_retry(ctx: Dict[str, Any]) -> str:
    """Determine if we should retry based on error type.

    Args:
        ctx: Workflow context with:
            - error (str): Error message
            - retry_count (int): Current retry count
            - max_retries (int): Maximum allowed retries

    Returns:
        "retry" or "fail"
    """
    error = ctx.get("error", "")
    retry_count = ctx.get("retry_count", 0)
    max_retries = ctx.get("max_retries", 3)

    # Don't retry on certain errors
    if "permission denied" in error.lower():
        return "fail"

    if retry_count >= max_retries:
        logger.info(f"Max retries ({max_retries}) reached")
        return "fail"

    return "retry"


def complexity_level(ctx: Dict[str, Any]) -> str:
    """Assess task complexity for resource allocation.

    Returns:
        "simple", "moderate", or "complex"
    """
    files_count = ctx.get("files_to_change", 1)
    lines_estimate = ctx.get("estimated_lines", 0)
    has_tests = ctx.get("requires_tests", False)

    if files_count > 10 or lines_estimate > 500:
        return "complex"

    if files_count > 3 or lines_estimate > 100 or has_tests:
        return "moderate"

    return "simple"


# =============================================================================
# Transform Functions
# =============================================================================

def prepare_report_data(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare data for report generation.

    Returns:
        Updated context with formatted report data
    """
    raw_findings = ctx.get("findings", [])

    # Group by severity
    critical = [f for f in raw_findings if f.get("severity") == "critical"]
    high = [f for f in raw_findings if f.get("severity") == "high"]
    medium = [f for f in raw_findings if f.get("severity") == "medium"]
    low = [f for f in raw_findings if f.get("severity") == "low"]

    return {
        "report_data": {
            "total_findings": len(raw_findings),
            "critical_count": len(critical),
            "high_count": len(high),
            "medium_count": len(medium),
            "low_count": len(low),
            "critical_findings": critical,
            "high_findings": high,
            "needs_attention": len(critical) > 0 or len(high) > 2,
        }
    }


# =============================================================================
# Registry Exports
# =============================================================================

# These are automatically loaded by the YAML workflow system
CONDITIONS = {
    "should_retry": should_retry,
    "complexity_level": complexity_level,
}

TRANSFORMS = {
    "prepare_report_data": prepare_report_data,
}
```

### Referencing from YAML

```yaml
workflows:
  smart_workflow:
    nodes:
      - id: assess_complexity
        type: condition
        condition: "complexity_level"      # References escape hatch
        branches:
          simple: quick_fix
          moderate: standard_process
          complex: detailed_analysis

      - id: prepare_for_report
        type: transform
        transform: "prepare_report_data"   # References escape hatch
        next:
          - generate_report
```

---

## 7. Testing Workflows

### Validate Workflow Structure

```bash
# Validate YAML syntax and structure
victor workflow validate workflows/code_review.yaml

# Validate with escape hatches
victor workflow validate workflows/code_review.yaml \
  --escape-hatches victor/coding/escape_hatches.py
```

### Run Workflows

```bash
# Run a specific workflow
victor workflow run workflows/code_review.yaml \
  --workflow parallel_review \
  --input '{"target_file": "src/app.py"}'

# Run with verbose output
victor workflow run workflows/code_review.yaml \
  --workflow parallel_review \
  --verbose

# Run in dry-run mode (no actual execution)
victor workflow run workflows/code_review.yaml \
  --workflow parallel_review \
  --dry-run
```

### Debugging Tips

1. **Use verbose mode**: Add `--verbose` to see step-by-step execution

2. **Check node results**: Access `_node_results` in the final state:
   ```python
   result = await graph.invoke(initial_state)
   for node_id, node_result in result.state.get("_node_results", {}).items():
       print(f"{node_id}: success={node_result.success}")
   ```

3. **Enable logging**:
   ```python
   import logging
   logging.getLogger("victor.workflows").setLevel(logging.DEBUG)
   ```

4. **Test escape hatches independently**:
   ```python
   from escape_hatches import code_quality_check

   # Test with mock context
   test_ctx = {"analysis": {"errors": 0, "warnings": 2}}
   result = code_quality_check(test_ctx)
   assert result == "good"
   ```

5. **Use auto-approve mode for testing**:
   ```python
   from victor.workflows.hitl import HITLMode, HITLExecutor

   executor = HITLExecutor(mode=HITLMode.AUTO_APPROVE)
   ```

---

## 8. Complete Example: Full Code Review Workflow

Here is a complete, production-ready code review workflow combining all concepts:

```yaml
# workflows/complete_code_review.yaml
#
# A comprehensive code review workflow that:
# 1. Analyzes code for issues
# 2. Runs parallel security and style checks
# 3. Routes based on quality assessment
# 4. Requires human approval for critical issues
# 5. Generates a detailed report

workflows:
  complete_code_review:
    description: "Comprehensive code review with parallel analysis and human gates"
    metadata:
      category: code_quality
      version: "2.0"
      author: "victor_team"
      requires_approval: true

    # Workflow-level settings
    max_execution_timeout_seconds: 1800    # 30 minute overall timeout
    default_node_timeout_seconds: 300      # 5 minute default per node
    max_iterations: 10                     # Max loop iterations

    nodes:
      # =====================================================
      # Phase 1: Initial Analysis
      # =====================================================
      - id: gather_context
        type: agent
        role: researcher
        goal: "Gather context about the code changes: files modified, scope of changes, and related components"
        tool_budget: 10
        allowed_tools:
          - read
          - grep
          - git
          - code_search
        output_key: change_context
        next:
          - parallel_analysis

      # =====================================================
      # Phase 2: Parallel Analysis
      # =====================================================
      - id: parallel_analysis
        type: parallel
        parallel_nodes:
          - security_scan
          - style_analysis
          - complexity_check
          - test_coverage_check
        join_strategy: all
        next:
          - merge_analysis

      - id: security_scan
        type: agent
        role: security_analyst
        goal: "Scan code for security vulnerabilities: SQL injection, XSS, hardcoded secrets, unsafe operations"
        tool_budget: 15
        allowed_tools:
          - read
          - grep
          - code_search
        output_key: security_findings

      - id: style_analysis
        type: agent
        role: style_reviewer
        goal: "Check code style, naming conventions, documentation, and PEP8 compliance"
        tool_budget: 10
        allowed_tools:
          - read
          - shell
        output_key: style_findings

      - id: complexity_check
        type: agent
        role: analyst
        goal: "Analyze cyclomatic complexity, function length, and code maintainability"
        tool_budget: 10
        allowed_tools:
          - read
          - symbols
        output_key: complexity_findings

      - id: test_coverage_check
        type: agent
        role: tester
        goal: "Verify test coverage for changed code and identify missing tests"
        tool_budget: 10
        allowed_tools:
          - read
          - grep
          - shell
        output_key: coverage_findings

      # =====================================================
      # Phase 3: Merge and Assess
      # =====================================================
      - id: merge_analysis
        type: transform
        transform: "merge_code_analysis"
        next:
          - assess_quality

      - id: assess_quality
        type: condition
        condition: "code_quality_check"
        branches:
          excellent: prepare_approval
          good: prepare_approval
          acceptable: minor_fixes_needed
          needs_improvement: major_fixes_needed

      # =====================================================
      # Phase 4: Fix Routes
      # =====================================================
      - id: minor_fixes_needed
        type: agent
        role: fixer
        goal: "Apply minor fixes for style issues and small improvements"
        tool_budget: 15
        allowed_tools:
          - read
          - edit
          - shell
        output_key: minor_fix_results
        next:
          - verify_fixes

      - id: major_fixes_needed
        type: agent
        role: fixer
        goal: "Address major code quality issues: security vulnerabilities, complexity problems, missing tests"
        tool_budget: 30
        allowed_tools:
          - read
          - write
          - edit
          - shell
        output_key: major_fix_results
        next:
          - verify_fixes

      - id: verify_fixes
        type: agent
        role: verifier
        goal: "Verify that applied fixes resolved the identified issues"
        tool_budget: 10
        allowed_tools:
          - read
          - shell
          - grep
        output_key: verification_results
        next:
          - check_verification

      - id: check_verification
        type: condition
        condition: "tests_passed"
        branches:
          "true": prepare_approval
          "false": escalate_issues

      - id: escalate_issues
        type: hitl
        hitl_type: review
        prompt: "Automated fixes could not resolve all issues. Please review and provide guidance."
        context_keys:
          - verification_results
          - major_fix_results
          - security_findings
        timeout: 900
        fallback: abort
        next:
          - prepare_approval

      # =====================================================
      # Phase 5: Human Approval
      # =====================================================
      - id: prepare_approval
        type: transform
        transform: "prepare_report_data"
        next:
          - human_approval

      - id: human_approval
        type: hitl
        hitl_type: approval
        prompt: |
          Code Review Summary
          ===================
          Please review the analysis results and approve or reject the changes.

          Critical issues will require explicit approval to proceed.
        context_keys:
          - report_data
          - change_context
          - security_findings
        timeout: 600
        fallback: abort
        next:
          - generate_final_report

      # =====================================================
      # Phase 6: Final Report
      # =====================================================
      - id: generate_final_report
        type: agent
        role: reporter
        goal: "Generate a comprehensive code review report with all findings, fixes applied, and recommendations"
        tool_budget: 5
        allowed_tools:
          - read
        output_key: final_report
```

### Supporting escape_hatches.py

```python
# escape_hatches.py for complete_code_review workflow
"""Escape hatches for the complete code review workflow."""

from typing import Any, Dict


def code_quality_check(ctx: Dict[str, Any]) -> str:
    """Assess overall code quality.

    Returns: "excellent", "good", "acceptable", or "needs_improvement"
    """
    # Get merged analysis results
    merged = ctx.get("merged_analysis", {})

    security = merged.get("security_findings", {})
    style = merged.get("style_findings", {})
    complexity = merged.get("complexity_findings", {})

    # Count issues by severity
    critical_issues = security.get("critical", 0)
    high_issues = security.get("high", 0)
    style_errors = style.get("errors", 0)
    complexity_score = complexity.get("average_complexity", 5)

    # Scoring logic
    if critical_issues > 0:
        return "needs_improvement"

    if high_issues > 0 or style_errors > 5 or complexity_score > 15:
        return "needs_improvement"

    if high_issues == 0 and style_errors <= 2 and complexity_score <= 8:
        return "excellent"

    if style_errors <= 3 and complexity_score <= 10:
        return "good"

    return "acceptable"


def tests_passed(ctx: Dict[str, Any]) -> str:
    """Check if verification tests passed."""
    results = ctx.get("verification_results", {})
    success = results.get("success", False)
    return "true" if success else "false"


def merge_code_analysis(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Merge all analysis results into unified structure."""
    return {
        "merged_analysis": {
            "security_findings": ctx.get("security_findings", {}),
            "style_findings": ctx.get("style_findings", {}),
            "complexity_findings": ctx.get("complexity_findings", {}),
            "coverage_findings": ctx.get("coverage_findings", {}),
        }
    }


def prepare_report_data(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare data for the final report."""
    merged = ctx.get("merged_analysis", {})

    security = merged.get("security_findings", {})
    style = merged.get("style_findings", {})

    return {
        "report_data": {
            "security_issues": security.get("issues", []),
            "style_issues": style.get("issues", []),
            "overall_score": _calculate_score(merged),
            "recommendation": _get_recommendation(merged),
        }
    }


def _calculate_score(merged: Dict[str, Any]) -> int:
    """Calculate overall score out of 100."""
    security = merged.get("security_findings", {})
    style = merged.get("style_findings", {})
    complexity = merged.get("complexity_findings", {})

    # Start at 100, deduct for issues
    score = 100
    score -= security.get("critical", 0) * 25
    score -= security.get("high", 0) * 10
    score -= style.get("errors", 0) * 3
    score -= max(0, complexity.get("average_complexity", 5) - 10)

    return max(0, score)


def _get_recommendation(merged: Dict[str, Any]) -> str:
    """Generate recommendation based on analysis."""
    score = _calculate_score(merged)

    if score >= 90:
        return "Ready to merge"
    if score >= 70:
        return "Minor changes recommended before merge"
    if score >= 50:
        return "Significant changes required"
    return "Major rework needed"


CONDITIONS = {
    "code_quality_check": code_quality_check,
    "tests_passed": tests_passed,
}

TRANSFORMS = {
    "merge_code_analysis": merge_code_analysis,
    "prepare_report_data": prepare_report_data,
}
```

### Running the Complete Workflow

```bash
# Validate
victor workflow validate workflows/complete_code_review.yaml \
  --escape-hatches escape_hatches.py

# Run
victor workflow run workflows/complete_code_review.yaml \
  --workflow complete_code_review \
  --input '{"target_files": ["src/api/routes.py", "src/api/handlers.py"]}'
```

---

## Next Steps

Now that you understand Victor workflows, explore:

- **[Vertical Development Guide](../development/extending/verticals.md)**: Create custom verticals with workflows
- **[StateGraph DSL](../api-reference/workflows.md)**: Build workflows programmatically in Python
- **[Multi-Agent Teams](../guides/MULTI_AGENT_TEAMS.md)**: Coordinate multiple agents within workflows
- **[Workflow Scheduling](../guides/WORKFLOW_SCHEDULER.md)**: Run workflows on schedules and triggers

## Reference

- **Node Types**: [docs/reference/workflow-nodes.md](../user-guide/yaml_workflow_syntax.md)
- **Escape Hatches**: [victor/coding/escape_hatches.py](https://github.com/anvai-labs/victor/blob/develop/victor/workflows/escape_hatches.py)
- **Example Workflows**: [victor/workflows/mode_workflows.yaml](https://github.com/anvai-labs/victor/blob/develop/victor/workflows/mode_workflows.yaml)
- **API Reference**: [docs/api-reference/workflows.md](../api-reference/workflows.md)
