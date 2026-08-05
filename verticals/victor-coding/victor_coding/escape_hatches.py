# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Escape hatches for Coding YAML workflows.

Complex conditions and transforms that cannot be expressed in YAML.
These are registered with the YAML workflow loader for use in condition nodes.

Example YAML usage:
    - id: check_tests
      type: condition
      condition: "tests_passing"  # References escape hatch
      branches:
        "passing": deploy
        "failing": fix_code
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


# =============================================================================
# Condition Functions
# =============================================================================


def tests_passing(ctx: Dict[str, Any]) -> str:
    """Check if tests are passing.

    Args:
        ctx: Workflow context with keys:
            - test_results (dict): Test execution results
            - min_coverage (float): Minimum coverage threshold

    Returns:
        "passing", "failing", or "no_tests"
    """
    test_results = ctx.get("test_results", {})
    min_coverage = ctx.get("min_coverage", 0.8)

    if not test_results:
        return "no_tests"

    passed = test_results.get("passed", 0)
    failed = test_results.get("failed", 0)
    coverage = test_results.get("coverage", 0)

    if failed > 0:
        return "failing"

    if coverage < min_coverage:
        return "failing"

    if passed > 0:
        return "passing"

    return "no_tests"


def code_quality_check(ctx: Dict[str, Any]) -> str:
    """Assess code quality based on linting and static analysis.

    Args:
        ctx: Workflow context with keys:
            - lint_results (dict): Linter output
            - type_check_results (dict): Type checker output
            - quality_threshold (str): Minimum quality level

    Returns:
        "excellent", "good", "acceptable", or "needs_improvement"
    """
    lint_results = ctx.get("lint_results", {})
    type_check_results = ctx.get("type_check_results", {})

    lint_errors = lint_results.get("errors", 0)
    lint_warnings = lint_results.get("warnings", 0)
    type_errors = type_check_results.get("errors", 0)

    if lint_errors == 0 and type_errors == 0 and lint_warnings == 0:
        return "excellent"

    if lint_errors == 0 and type_errors == 0:
        return "good"

    if lint_errors <= 3 and type_errors <= 2:
        return "acceptable"

    return "needs_improvement"


def should_retry_implementation(ctx: Dict[str, Any]) -> str:
    """Determine if implementation should be retried.

    Args:
        ctx: Workflow context with keys:
            - test_results (dict): Test execution results
            - iteration_count (int): Current iteration
            - max_iterations (int): Maximum allowed iterations
            - error (str): Error message if any

    Returns:
        "retry" or "give_up"
    """
    iteration = ctx.get("iteration_count", 0)
    max_iter = ctx.get("max_iterations", 3)
    test_results = ctx.get("test_results", {})
    error = ctx.get("error")

    if iteration >= max_iter:
        logger.info(f"Max iterations ({max_iter}) reached, giving up")
        return "give_up"

    if error and "fatal" in error.lower():
        return "give_up"

    failed = test_results.get("failed", 0)
    if failed > 0:
        return "retry"

    return "give_up"


def review_verdict(ctx: Dict[str, Any]) -> str:
    """Determine code review verdict.

    Args:
        ctx: Workflow context with keys:
            - review_comments (list): Review comments
            - approval_status (str): Approval status
            - blocking_issues (int): Number of blocking issues

    Returns:
        "approved", "changes_requested", or "needs_discussion"
    """
    approval_status = ctx.get("approval_status", "pending")
    blocking_issues = ctx.get("blocking_issues", 0)
    review_comments = ctx.get("review_comments", [])

    if approval_status == "approved" and blocking_issues == 0:
        return "approved"

    if blocking_issues > 0:
        return "changes_requested"

    if len(review_comments) > 5:
        return "needs_discussion"

    if approval_status == "pending":
        return "needs_discussion"

    return "changes_requested"


def complexity_assessment(ctx: Dict[str, Any]) -> str:
    """Assess task complexity for planning.

    Args:
        ctx: Workflow context with keys:
            - files_to_modify (int): Number of files to change
            - estimated_lines (int): Estimated lines of code
            - dependencies (list): External dependencies involved

    Returns:
        "simple", "moderate", "complex", or "major"
    """
    files = ctx.get("files_to_modify", 1)
    lines = ctx.get("estimated_lines", 0)
    dependencies = ctx.get("dependencies", [])

    dep_count = len(dependencies) if isinstance(dependencies, list) else 0

    if files <= 1 and lines <= 50 and dep_count == 0:
        return "simple"

    if files <= 3 and lines <= 200:
        return "moderate"

    if files <= 10 or lines <= 500:
        return "complex"

    return "major"


def complexity_check(ctx: Dict[str, Any]) -> str:
    """Assess task complexity from task analysis for team routing.

    Used by team_node workflows to route tasks to appropriate team sizes.
    Evaluates task_analysis output to determine complexity level.

    Args:
        ctx: Workflow context with keys:
            - task_analysis (str|dict): Task analysis from planner agent
            - user_task (str): Original user task description

    Returns:
        "simple", "medium", or "complex"
    """
    task_analysis = ctx.get("task_analysis", "")
    user_task = ctx.get("user_task", "")

    # Handle string analysis (from agent output)
    if isinstance(task_analysis, str):
        analysis_lower = task_analysis.lower()

        # Check for explicit complexity mentions
        if any(kw in analysis_lower for kw in ["complex", "major", "significant", "large"]):
            return "complex"
        if any(kw in analysis_lower for kw in ["medium", "moderate", "several"]):
            return "medium"
        if any(kw in analysis_lower for kw in ["simple", "trivial", "straightforward", "minor"]):
            return "simple"

        # Estimate from team size mentions
        if "team size: 4" in analysis_lower or "team size: 3" in analysis_lower:
            return "complex"
        if "team size: 2" in analysis_lower:
            return "medium"
        if "team size: 1" in analysis_lower:
            return "simple"

    # Handle dict analysis
    elif isinstance(task_analysis, dict):
        complexity = task_analysis.get("complexity", "").lower()
        if complexity in ["complex", "major"]:
            return "complex"
        if complexity in ["medium", "moderate"]:
            return "medium"
        if complexity in ["simple", "trivial"]:
            return "simple"

        # Check team size from dict
        team_size = task_analysis.get("team_size", 1)
        if isinstance(team_size, int):
            if team_size >= 4:
                return "complex"
            if team_size >= 2:
                return "medium"
            return "simple"

    # Fallback: estimate from user task length/keywords
    task_lower = user_task.lower() if isinstance(user_task, str) else ""
    if len(task_lower) > 200 or any(
        kw in task_lower for kw in ["refactor", "redesign", "migrate", "overhaul"]
    ):
        return "complex"
    if len(task_lower) > 100 or any(
        kw in task_lower for kw in ["add feature", "implement", "create"]
    ):
        return "medium"

    return "simple"


def tdd_cycle_status(ctx: Dict[str, Any]) -> str:
    """Determine TDD cycle status.

    Args:
        ctx: Workflow context with keys:
            - tests_written (bool): Whether tests are written
            - tests_passing (bool): Whether tests pass
            - implementation_complete (bool): Whether implementation is done

    Returns:
        "red", "green", or "refactor"
    """
    tests_written = ctx.get("tests_written", False)
    passing = ctx.get("tests_passing", False)
    impl_complete = ctx.get("implementation_complete", False)

    if not tests_written:
        return "red"

    if not passing:
        return "red"

    if passing and impl_complete:
        return "refactor"

    return "green"


def bugfix_priority(ctx: Dict[str, Any]) -> str:
    """Determine bugfix priority level.

    Args:
        ctx: Workflow context with keys:
            - severity (str): Bug severity (critical, high, medium, low)
            - affected_users (int): Number of affected users
            - has_workaround (bool): Whether workaround exists

    Returns:
        "p0", "p1", "p2", or "p3"
    """
    severity = ctx.get("severity", "medium")
    affected_users = ctx.get("affected_users", 0)
    has_workaround = ctx.get("has_workaround", False)

    if severity == "critical":
        return "p0"

    if severity == "high" and affected_users > 100:
        return "p0"

    if severity == "high" or affected_users > 50:
        return "p1"

    if severity == "medium" and not has_workaround:
        return "p2"

    return "p3"


def should_continue_fixing(ctx: Dict[str, Any]) -> str:
    """Determine if agent should continue attempting fixes.

    Multi-factor decision based on iteration count, error patterns, and progress.

    Args:
        ctx: Workflow context with keys:
            - fix_iterations (int): Number of fix attempts made
            - max_iterations (int): Maximum allowed iterations (default 5)
            - progress_made (bool): Whether last iteration made progress
            - test_results (dict): Current test results

    Returns:
        "continue_fixing", "escalate", or "submit_best_effort"
    """
    iterations = ctx.get("fix_iterations", 0)
    max_iter = ctx.get("max_iterations", 5)
    progress_made = ctx.get("progress_made", False)
    test_results = ctx.get("test_results", {})

    # Calculate pass rate
    passed = test_results.get("passed", 0)
    failed = test_results.get("failed", 0)
    total = passed + failed
    pass_rate = passed / total if total > 0 else 0

    # Max iterations reached
    if iterations >= max_iter:
        logger.info(f"Max fix iterations ({max_iter}) reached, submitting best effort")
        return "submit_best_effort"

    # High pass rate achieved
    if pass_rate >= 0.95:
        return "submit_best_effort"

    # Still making progress
    if progress_made and iterations < max_iter:
        return "continue_fixing"

    # No progress after several attempts
    if iterations >= 3 and not progress_made:
        return "escalate"

    return "continue_fixing"


# =============================================================================
# Transform Functions
# =============================================================================


def merge_code_analysis(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Merge results from parallel code analysis operations.

    Args:
        ctx: Workflow context with parallel analysis results

    Returns:
        Merged analysis results
    """
    ast_analysis = ctx.get("ast_analysis", {})
    semantic_analysis = ctx.get("semantic_analysis", {})
    dependency_analysis = ctx.get("dependency_analysis", {})

    issues = []
    issues.extend(ast_analysis.get("issues", []))
    issues.extend(semantic_analysis.get("issues", []))
    issues.extend(dependency_analysis.get("issues", []))

    return {
        "total_issues": len(issues),
        "issues": issues,
        "complexity_score": ast_analysis.get("complexity", 0),
        "dependencies": dependency_analysis.get("dependencies", []),
        "symbols": semantic_analysis.get("symbols", []),
    }


def format_implementation_plan(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Format implementation plan from analysis.

    Args:
        ctx: Workflow context with research_findings

    Returns:
        Formatted implementation plan
    """
    findings = ctx.get("research_findings", {})
    task = ctx.get("task", "")

    files_to_modify = findings.get("files_to_modify", [])
    approach = findings.get("approach", "")
    risks = findings.get("risks", [])

    steps = []
    for i, file in enumerate(files_to_modify, 1):
        steps.append(
            {
                "step": i,
                "file": file,
                "action": "modify",
            }
        )

    return {
        "task": task,
        "steps": steps,
        "approach": approach,
        "risks": risks,
        "estimated_files": len(files_to_modify),
    }


# =============================================================================
# Registry Exports
# =============================================================================

# Conditions available in YAML workflows
CONDITIONS = {
    "tests_passing": tests_passing,
    "code_quality_check": code_quality_check,
    "should_retry_implementation": should_retry_implementation,
    "review_verdict": review_verdict,
    "complexity_assessment": complexity_assessment,
    "complexity_check": complexity_check,
    "tdd_cycle_status": tdd_cycle_status,
    "bugfix_priority": bugfix_priority,
    "should_continue_fixing": should_continue_fixing,
}

# Transforms available in YAML workflows
TRANSFORMS = {
    "merge_code_analysis": merge_code_analysis,
    "format_implementation_plan": format_implementation_plan,
}

__all__ = [
    # Conditions
    "tests_passing",
    "code_quality_check",
    "should_retry_implementation",
    "review_verdict",
    "complexity_assessment",
    "complexity_check",
    "tdd_cycle_status",
    "bugfix_priority",
    "should_continue_fixing",
    # Transforms
    "merge_code_analysis",
    "format_implementation_plan",
    # Registries
    "CONDITIONS",
    "TRANSFORMS",
]
