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

"""Escape hatches for Benchmark YAML workflows.

Complex conditions and transforms that cannot be expressed in YAML.
These are registered with the YAML workflow loader for use in condition nodes.

Example YAML usage:
    - id: check_solution
      type: condition
      condition: "solution_quality_check"  # References escape hatch
      branches:
        "high_quality": finalize
        "needs_improvement": refine
        "failed": abort
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Literal

if TYPE_CHECKING:
    from victor.tools.registry import ToolRegistry
    from victor.workflows.definition import ComputeNode
    from victor.workflows.context import WorkflowContext
    from victor_contracts.workflows import NodeResult, ExecutorNodeStatus

logger = logging.getLogger(__name__)


# =============================================================================
# Enums for Type-Safe Return Values
# =============================================================================


class SolutionQuality(str, Enum):
    """Quality assessment of a benchmark solution."""

    HIGH_QUALITY = "high_quality"  # Solution passes all quality thresholds
    NEEDS_IMPROVEMENT = "needs_improvement"  # Solution passes but has quality issues
    FAILED = "failed"  # Solution does not meet minimum requirements


class CoverageStatus(str, Enum):
    """Test coverage assessment for a solution."""

    SUFFICIENT = "sufficient"  # Coverage meets all requirements
    NEEDS_MORE_TESTS = "needs_more_tests"  # Coverage below thresholds


class ComplexityLevel(str, Enum):
    """Complexity assessment of a problem/solution."""

    SIMPLE = "simple"  # Low complexity, straightforward solution
    MEDIUM = "medium"  # Moderate complexity, some challenges
    COMPLEX = "complex"  # High complexity, requires careful approach


# =============================================================================
# Condition Functions
# =============================================================================


def solution_quality_check(ctx: Dict[str, Any]) -> SolutionQuality:
    """Check solution quality based on multiple factors.

    Evaluates test pass rate, code quality metrics, and solution completeness.

    Args:
        ctx: Workflow context with keys:
            - test_results (dict): Test execution results
            - code_quality (dict): Code quality metrics (linting, complexity)
            - solution_completeness (float): How complete the solution is (0-1)
            - pass_threshold (float): Minimum pass rate required (default 0.8)

    Returns:
        SolutionQuality enum value: HIGH_QUALITY, NEEDS_IMPROVEMENT, or FAILED
    """
    test_results = ctx.get("test_results", {})
    code_quality = ctx.get("code_quality", {})
    completeness = ctx.get("solution_completeness", 0)
    pass_threshold = ctx.get("pass_threshold", 0.8)

    # Calculate test pass rate
    tests_passed = test_results.get("passed", 0)
    tests_total = test_results.get("total", 0)
    pass_rate = tests_passed / tests_total if tests_total > 0 else 0

    # Check for critical failures
    if test_results.get("error"):
        logger.warning(f"Solution failed: {test_results.get('error')}")
        return SolutionQuality.FAILED

    if pass_rate < 0.3:
        return SolutionQuality.FAILED

    # Quality factors
    lint_score = code_quality.get("lint_score", 1.0)
    complexity_ok = code_quality.get("complexity_acceptable", True)

    # High quality: good pass rate, clean code, complete solution
    if pass_rate >= pass_threshold and lint_score >= 0.9 and completeness >= 0.9:
        if complexity_ok:
            return SolutionQuality.HIGH_QUALITY

    # Needs improvement: passing but could be better
    if pass_rate >= 0.5 or completeness >= 0.5:
        return SolutionQuality.NEEDS_IMPROVEMENT

    return SolutionQuality.FAILED


def test_coverage_check(ctx: Dict[str, Any]) -> CoverageStatus:
    """Check if test coverage is sufficient for the solution.

    Evaluates line coverage, branch coverage, and edge case handling.

    Args:
        ctx: Workflow context with keys:
            - coverage_report (dict): Coverage report with line/branch data
            - min_line_coverage (float): Minimum line coverage (default 0.7)
            - min_branch_coverage (float): Minimum branch coverage (default 0.6)
            - edge_cases_covered (list): List of covered edge cases

    Returns:
        CoverageStatus enum value: SUFFICIENT or NEEDS_MORE_TESTS
    """
    coverage_report = ctx.get("coverage_report", {})
    min_line = ctx.get("min_line_coverage", 0.7)
    min_branch = ctx.get("min_branch_coverage", 0.6)
    edge_cases = ctx.get("edge_cases_covered", [])

    line_coverage = coverage_report.get("line_coverage", 0)
    branch_coverage = coverage_report.get("branch_coverage", 0)
    required_edge_cases = ctx.get("required_edge_cases", [])

    # Check line coverage
    if line_coverage < min_line:
        logger.info(f"Line coverage {line_coverage:.1%} below threshold {min_line:.1%}")
        return CoverageStatus.NEEDS_MORE_TESTS

    # Check branch coverage
    if branch_coverage < min_branch:
        logger.info(f"Branch coverage {branch_coverage:.1%} below threshold {min_branch:.1%}")
        return CoverageStatus.NEEDS_MORE_TESTS

    # Check edge case coverage
    if required_edge_cases:
        covered_set = set(edge_cases) if isinstance(edge_cases, list) else set()
        required_set = set(required_edge_cases)
        missing = required_set - covered_set

        if len(missing) > len(required_set) * 0.3:  # Allow 30% missing
            logger.info(f"Missing edge cases: {missing}")
            return CoverageStatus.NEEDS_MORE_TESTS

    return CoverageStatus.SUFFICIENT


def complexity_check(ctx: Dict[str, Any]) -> ComplexityLevel:
    """Assess problem complexity to adjust strategy.

    Evaluates code size, cyclomatic complexity, and problem domain.

    Args:
        ctx: Workflow context with keys:
            - problem (dict): Problem definition with metadata
            - code_metrics (dict): Code complexity metrics
            - estimated_loc (int): Estimated lines of code needed
            - domain (str): Problem domain (e.g., "algorithm", "system")

    Returns:
        ComplexityLevel enum value: SIMPLE, MEDIUM, or COMPLEX
    """
    problem = ctx.get("problem", {})
    code_metrics = ctx.get("code_metrics", {})
    estimated_loc = ctx.get("estimated_loc", 0)
    domain = ctx.get("domain", "")

    # Check explicit difficulty if provided
    difficulty = problem.get("difficulty", "").lower()
    if difficulty in ["easy", "trivial"]:
        return ComplexityLevel.SIMPLE
    if difficulty in ["hard", "expert", "advanced"]:
        return ComplexityLevel.COMPLEX

    # Check code complexity metrics
    cyclomatic = code_metrics.get("cyclomatic_complexity", 0)
    if cyclomatic > 20:
        return ComplexityLevel.COMPLEX
    if cyclomatic > 10:
        return ComplexityLevel.MEDIUM

    # Check estimated size
    if estimated_loc > 500:
        return ComplexityLevel.COMPLEX
    if estimated_loc > 100:
        return ComplexityLevel.MEDIUM

    # Check domain complexity
    complex_domains = ["distributed", "concurrent", "ml", "compiler", "database"]
    if domain.lower() in complex_domains:
        return ComplexityLevel.COMPLEX

    medium_domains = ["algorithm", "system", "network"]
    if domain.lower() in medium_domains:
        return ComplexityLevel.MEDIUM

    return ComplexityLevel.SIMPLE


class TestExecutionStatus(str, Enum):
    """Test execution outcome status."""

    ALL_PASS = "all_pass"  # All tests passed
    PARTIAL_PASS = "partial_pass"  # Some tests passed
    ALL_FAIL = "all_fail"  # All tests failed
    ERROR = "error"  # Test execution error


class FixingDecision(str, Enum):
    """Decision on whether to continue fixing."""

    CONTINUE_FIXING = "continue_fixing"  # Continue attempting fixes
    ESCALATE = "escalate"  # Escalate to human intervention
    SUBMIT_BEST_EFFORT = "submit_best_effort"  # Submit current best solution


class VerificationStatus(str, Enum):
    """Solution verification status."""

    VERIFIED = "verified"  # Solution fully verified
    PARTIAL = "partial"  # Partially verified
    FAILED = "failed"  # Verification failed


class EscalationDecision(str, Enum):
    """Human-in-the-loop escalation decision."""

    CONTINUE = "continue"  # Continue with automation
    SUBMIT = "submit"  # Submit current solution
    ABORT = "abort"  # Abort the task


class PasskProgress(str, Enum):
    """Pass@k generation progress status."""

    SUFFICIENT = "sufficient"  # Generated sufficient solutions
    NEED_MORE = "need_more"  # Need more solutions
    MAX_REACHED = "max_reached"  # Reached max iterations


# =============================================================================
# Transform Functions
# =============================================================================


def extract_patch(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Extract git diff/patch from generated code.

    Creates a unified diff format patch from the solution.

    Args:
        ctx: Workflow context with keys:
            - original_code (str): Original source code
            - modified_code (str): Modified source code
            - file_path (str): Path to the file being modified
            - commit_message (str): Optional commit message

    Returns:
        Dict with:
            - patch (str): Unified diff format patch
            - hunks (list): Individual diff hunks
            - stats (dict): Diff statistics (additions, deletions)
    """
    import difflib

    original = ctx.get("original_code", "")
    modified = ctx.get("modified_code", "")
    file_path = ctx.get("file_path", "solution.py")

    # Split into lines
    original_lines = original.splitlines(keepends=True)
    modified_lines = modified.splitlines(keepends=True)

    # Ensure lines end with newlines for proper diff
    if original_lines and not original_lines[-1].endswith("\n"):
        original_lines[-1] += "\n"
    if modified_lines and not modified_lines[-1].endswith("\n"):
        modified_lines[-1] += "\n"

    # Generate unified diff
    diff_lines = list(
        difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
        )
    )

    patch = "".join(diff_lines)

    # Calculate statistics
    additions = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
    deletions = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))

    # Extract hunks (sections of changes)
    hunks: List[str] = []
    current_hunk: List[str] = []
    for line in diff_lines:
        if line.startswith("@@"):
            if current_hunk:
                hunks.append("".join(current_hunk))
            current_hunk = [line]
        elif current_hunk:
            current_hunk.append(line)
    if current_hunk:
        hunks.append("".join(current_hunk))

    return {
        "patch": patch,
        "hunks": hunks,
        "stats": {
            "additions": additions,
            "deletions": deletions,
            "files_changed": 1 if patch else 0,
        },
        "file_path": file_path,
    }


def merge_tool_results(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Combine results from multiple tool calls.

    Merges outputs from parallel tool executions into a unified result.

    Args:
        ctx: Workflow context with keys:
            - tool_results (list): List of individual tool results
            - preserve_order (bool): Whether to maintain execution order
            - aggregate_errors (bool): Collect all errors vs. fail fast

    Returns:
        Dict with:
            - combined_output (dict): Merged tool outputs
            - success (bool): Overall success status
            - errors (list): Any errors encountered
            - execution_order (list): Order of tool execution
    """
    tool_results = ctx.get("tool_results", [])
    preserve_order = ctx.get("preserve_order", True)
    aggregate_errors = ctx.get("aggregate_errors", True)

    combined_output: Dict[str, Any] = {}
    errors: List[str] = []
    execution_order: List[str] = []
    all_success = True

    for result in tool_results:
        if not isinstance(result, dict):
            continue

        tool_name = result.get("tool_name", "unknown")
        execution_order.append(tool_name)

        # Check for errors
        if result.get("error"):
            errors.append(f"{tool_name}: {result['error']}")
            all_success = False
            if not aggregate_errors:
                break

        # Merge output
        output = result.get("output", {})
        if isinstance(output, dict):
            # Namespace by tool name to avoid collisions
            combined_output[tool_name] = output
        else:
            combined_output[tool_name] = {"value": output}

        # Track success
        if not result.get("success", True):
            all_success = False

    return {
        "combined_output": combined_output,
        "success": all_success and not errors,
        "errors": errors,
        "execution_order": (execution_order if preserve_order else sorted(execution_order)),
        "tool_count": len(tool_results),
    }


def format_for_evaluation(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare output for test runner evaluation.

    Formats the solution and metadata for benchmark evaluation harness.

    Args:
        ctx: Workflow context with keys:
            - solution_code (str): The generated solution code
            - patch (str): Git patch if available
            - problem_id (str): Benchmark problem identifier
            - model_info (dict): Model used for generation
            - generation_metadata (dict): Timing and token stats

    Returns:
        Dict with:
            - submission (dict): Formatted submission for evaluator
            - metadata (dict): Evaluation metadata
            - validation (dict): Pre-evaluation validation results
    """
    solution_code = ctx.get("solution_code", "")
    patch = ctx.get("patch", "")
    problem_id = ctx.get("problem_id", "")
    model_info = ctx.get("model_info", {})
    generation_metadata = ctx.get("generation_metadata", {})

    # Pre-validate the solution
    validation_errors: List[str] = []
    validation_warnings: List[str] = []

    if not solution_code and not patch:
        validation_errors.append("No solution code or patch provided")

    if solution_code:
        # Check for syntax errors
        try:
            compile(solution_code, "<solution>", "exec")
        except SyntaxError as e:
            validation_errors.append(f"Syntax error: {e}")

        # Check for common issues
        if "TODO" in solution_code or "FIXME" in solution_code:
            validation_warnings.append("Solution contains TODO/FIXME markers")

        if "pass" in solution_code and solution_code.strip().endswith("pass"):
            validation_warnings.append("Solution may have incomplete implementations")

    # Build submission
    submission = {
        "problem_id": problem_id,
        "solution": solution_code,
        "patch": patch,
        "format": "code" if solution_code else "patch",
    }

    # Build metadata
    metadata = {
        "model": model_info.get("model_name", "unknown"),
        "provider": model_info.get("provider", "unknown"),
        "generation_time": generation_metadata.get("duration_seconds", 0),
        "tokens_used": generation_metadata.get("total_tokens", 0),
        "attempts": generation_metadata.get("attempts", 1),
    }

    return {
        "submission": submission,
        "metadata": metadata,
        "validation": {
            "valid": len(validation_errors) == 0,
            "errors": validation_errors,
            "warnings": validation_warnings,
        },
    }


# =============================================================================
# Handler Classes
# =============================================================================


@dataclass
class RunTestsHandler:
    """Execute tests and return results.

    Runs the test suite against the generated solution.

    Example YAML:
        - id: run_tests
          type: compute
          handler: run_tests
          inputs:
            test_file: tests/test_solution.py
            solution_file: solution.py
            timeout: 60
          output: test_results
    """

    default_timeout: int = 60

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        from victor_contracts.workflows import NodeResult, ExecutorNodeStatus

        start_time = time.time()

        test_file = node.input_mapping.get("test_file", "")
        timeout = node.input_mapping.get("timeout", self.default_timeout)
        test_framework = node.input_mapping.get("framework", "pytest")

        if not test_file:
            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.FAILED,
                error="No test file specified",
                duration_seconds=time.time() - start_time,
            )

        # Build test command
        if test_framework == "pytest":
            cmd = f"python -m pytest {test_file} -v --tb=short --timeout={timeout}"
        elif test_framework == "unittest":
            cmd = f"python -m unittest {test_file} -v"
        else:
            cmd = f"python {test_file}"

        try:
            result = await tool_registry.execute("shell", command=cmd, timeout=timeout + 10)

            # Parse test output
            output_text = result.output if hasattr(result, "output") else str(result)
            test_output = self._parse_test_output(output_text, test_framework)

            output = {
                "success": result.success if hasattr(result, "success") else False,
                "passed": test_output.get("passed", 0),
                "failed": test_output.get("failed", 0),
                "total": test_output.get("total", 0),
                "errors": test_output.get("errors", []),
                "raw_output": output_text,
            }

            output_key = node.output_key or node.id
            context.set(output_key, output)

            status = (
                ExecutorNodeStatus.COMPLETED if output["success"] else ExecutorNodeStatus.FAILED
            )

            return NodeResult(
                node_id=node.id,
                status=status,
                output=output,
                duration_seconds=time.time() - start_time,
                tool_calls_used=1,
            )

        except Exception as e:
            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.FAILED,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )

    def _parse_test_output(self, output: str, framework: str) -> Dict[str, Any]:
        """Parse test output to extract results."""
        result: Dict[str, Any] = {"passed": 0, "failed": 0, "total": 0, "errors": []}

        if framework == "pytest":
            # Parse pytest output: "5 passed, 2 failed"
            match = re.search(r"(\d+) passed", output)
            if match:
                result["passed"] = int(match.group(1))

            match = re.search(r"(\d+) failed", output)
            if match:
                result["failed"] = int(match.group(1))

            match = re.search(r"(\d+) error", output)
            if match:
                result["errors"].append(f"{match.group(1)} errors")

        elif framework == "unittest":
            # Parse unittest output: "Ran 10 tests"
            match = re.search(r"Ran (\d+) test", output)
            if match:
                result["total"] = int(match.group(1))

            if "OK" in output:
                result["passed"] = result["total"]
            elif "FAILED" in output:
                match = re.search(r"failures=(\d+)", output)
                if match:
                    result["failed"] = int(match.group(1))
                result["passed"] = result["total"] - result["failed"]

        result["total"] = result["passed"] + result["failed"]
        return result


@dataclass
class ValidatePatchHandler:
    """Validate patch syntax and applicability.

    Checks that a patch is well-formed and can be applied.

    Example YAML:
        - id: validate_patch
          type: compute
          handler: validate_patch
          inputs:
            patch: $ctx.generated_patch
            target_file: src/solution.py
          output: validation_result
    """

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        from victor_contracts.workflows import NodeResult, ExecutorNodeStatus

        start_time = time.time()

        patch = node.input_mapping.get("patch", "")
        target_file = node.input_mapping.get("target_file", "")

        if not patch:
            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.FAILED,
                error="No patch provided",
                duration_seconds=time.time() - start_time,
            )

        validation_result = self._validate_patch_syntax(patch)

        if not validation_result["valid"]:
            output_key = node.output_key or node.id
            context.set(output_key, validation_result)
            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.FAILED,
                output=validation_result,
                error=validation_result.get("error", "Invalid patch"),
                duration_seconds=time.time() - start_time,
            )

        # If target file specified, try a dry-run apply
        if target_file:
            try:
                result = await tool_registry.execute(
                    "shell",
                    command=f"echo '{patch}' | git apply --check -",
                )
                validation_result["applicable"] = (
                    result.success if hasattr(result, "success") else False
                )
                if not validation_result["applicable"]:
                    validation_result["apply_error"] = (
                        result.output if hasattr(result, "output") else "Unknown error"
                    )
            except Exception as e:
                validation_result["applicable"] = False
                validation_result["apply_error"] = str(e)

        output_key = node.output_key or node.id
        context.set(output_key, validation_result)

        return NodeResult(
            node_id=node.id,
            status=ExecutorNodeStatus.COMPLETED,
            output=validation_result,
            duration_seconds=time.time() - start_time,
            tool_calls_used=1 if target_file else 0,
        )

    def _validate_patch_syntax(self, patch: str) -> Dict[str, Any]:
        """Validate patch syntax without applying."""
        result: Dict[str, Any] = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {"additions": 0, "deletions": 0, "hunks": 0},
        }

        lines = patch.split("\n")

        # Check for basic patch structure
        has_header = False
        has_hunks = False

        for line in lines:
            if line.startswith("---") or line.startswith("+++"):
                has_header = True
            if line.startswith("@@"):
                has_hunks = True
                result["stats"]["hunks"] += 1
            if line.startswith("+") and not line.startswith("+++"):
                result["stats"]["additions"] += 1
            if line.startswith("-") and not line.startswith("---"):
                result["stats"]["deletions"] += 1

        if not has_header:
            result["warnings"].append("Missing file header (--- / +++)")

        if not has_hunks:
            result["errors"].append("No hunks found (missing @@ markers)")
            result["valid"] = False

        # Check hunk headers are valid
        hunk_pattern = re.compile(r"^@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@")
        for line in lines:
            if line.startswith("@@"):
                if not hunk_pattern.match(line):
                    result["errors"].append(f"Invalid hunk header: {line}")
                    result["valid"] = False

        return result


# =============================================================================
# Additional Workflow Conditions
# =============================================================================


def test_execution_status(ctx: Dict[str, Any]) -> TestExecutionStatus:
    """Determine test execution outcome.

    Args:
        ctx: Workflow context with keys:
            - test_results (dict): Test results with passed/failed counts
            - exit_code (int): Test execution exit code
            - error_message (str): Any error message

    Returns:
        TestExecutionStatus enum value
    """
    exit_code = ctx.get("exit_code", 0)
    error_message = ctx.get("error_message", "")
    test_results = ctx.get("test_results", {})

    if error_message or exit_code < 0:
        return TestExecutionStatus.ERROR

    passed = test_results.get("passed", 0)
    failed = test_results.get("failed", 0)
    total = test_results.get("total", passed + failed)

    if total == 0:
        # No tests found - treat as error
        return TestExecutionStatus.ERROR

    if failed == 0 and passed > 0:
        return TestExecutionStatus.ALL_PASS

    if passed > 0:
        return TestExecutionStatus.PARTIAL_PASS

    return TestExecutionStatus.ALL_FAIL


def should_continue_fixing(ctx: Dict[str, Any]) -> FixingDecision:
    """Determine if agent should continue attempting fixes.

    Multi-factor decision based on iteration count, error patterns, and progress.

    Args:
        ctx: Workflow context with keys:
            - fix_iterations (int): Number of fix attempts made
            - max_iterations (int): Maximum allowed iterations (default 5)
            - progress_made (bool): Whether last iteration made progress
            - pass_rate (float): Current test pass rate

    Returns:
        FixingDecision enum value
    """
    iterations = ctx.get("fix_iterations", 0)
    max_iter = ctx.get("max_iterations", 5)
    progress_made = ctx.get("progress_made", False)
    pass_rate = ctx.get("pass_rate", 0)

    # Max iterations reached
    if iterations >= max_iter:
        logger.info(f"Max fix iterations ({max_iter}) reached, submitting best effort")
        return FixingDecision.SUBMIT_BEST_EFFORT

    # High pass rate achieved
    if pass_rate >= 0.95:
        return FixingDecision.SUBMIT_BEST_EFFORT

    # Still making progress
    if progress_made and iterations < max_iter:
        return FixingDecision.CONTINUE_FIXING

    # No progress after several attempts
    if iterations >= 3 and not progress_made:
        return FixingDecision.ESCALATE

    return FixingDecision.CONTINUE_FIXING


def code_complexity_check(ctx: Dict[str, Any]) -> str:
    """Assess code complexity to determine approach.

    Wrapper around complexity_check that uses the naming expected by workflows.

    Args:
        ctx: Workflow context with complexity indicators

    Returns:
        "simple", "medium", or "complex"
    """
    # Delegate to the existing complexity_check function
    result = complexity_check(ctx)
    # Map "medium" to "moderate" for workflow compatibility if needed
    return result


def verification_status(ctx: Dict[str, Any]) -> VerificationStatus:
    """Check verification results for solution.

    Args:
        ctx: Workflow context with keys:
            - syntax_valid (bool): Code syntax is valid
            - tests_pass (bool): Tests pass
            - lint_clean (bool): Linting passes

    Returns:
        VerificationStatus enum value
    """
    syntax_valid = ctx.get("syntax_valid", False)
    tests_pass = ctx.get("tests_pass", False)
    lint_clean = ctx.get("lint_clean", True)

    if not syntax_valid:
        return VerificationStatus.FAILED

    if tests_pass and lint_clean:
        return VerificationStatus.VERIFIED

    if tests_pass or syntax_valid:
        return VerificationStatus.PARTIAL

    return VerificationStatus.FAILED


def escalation_decision(ctx: Dict[str, Any]) -> EscalationDecision:
    """Handle escalation decision from HITL node.

    Args:
        ctx: Workflow context with keys:
            - hitl_response (str): User response from HITL

    Returns:
        EscalationDecision enum value
    """
    response = ctx.get("hitl_response", "").lower().strip()

    if "continue" in response or response == "1":
        return EscalationDecision.CONTINUE
    if "submit" in response or response == "2":
        return EscalationDecision.SUBMIT
    if "abort" in response or response == "3":
        return EscalationDecision.ABORT

    # Default to submit on timeout/fallback
    return EscalationDecision.SUBMIT


def passk_progress_check(ctx: Dict[str, Any]) -> PasskProgress:
    """Check pass@k generation progress.

    Determines if enough solutions have been generated or if more are needed.

    Args:
        ctx: Workflow context with keys:
            - valid_solutions (list): List of valid solutions generated
            - target_k (int): Target number of solutions (default 10)
            - generation_iterations (int): Number of generation rounds
            - max_iterations (int): Max generation rounds (default 3)
            - passed_solutions (int): Number of solutions that pass tests

    Returns:
        PasskProgress enum value
    """
    solutions = ctx.get("valid_solutions", [])
    target_k = ctx.get("target_k", 10)
    iterations = ctx.get("generation_iterations", 0)
    max_iter = ctx.get("max_iterations", 3)
    passed = ctx.get("passed_solutions", 0)

    solution_count = len(solutions) if isinstance(solutions, list) else 0

    # Max iterations reached
    if iterations >= max_iter:
        logger.info(f"Max generation iterations ({max_iter}) reached")
        return PasskProgress.MAX_REACHED

    # Have enough passing solutions for high confidence
    if passed >= target_k * 0.5:
        return PasskProgress.SUFFICIENT

    # Have target number of solutions
    if solution_count >= target_k:
        return PasskProgress.SUFFICIENT

    # Need more solutions
    return PasskProgress.NEED_MORE


# =============================================================================
# Additional Workflow Transforms
# =============================================================================


def extract_error_patterns(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Extract error patterns from test failures.

    Args:
        ctx: Workflow context with failure data

    Returns:
        Dict with categorized error patterns
    """
    failures = ctx.get("failures", [])
    test_results = ctx.get("test_results", {})
    error_output = test_results.get("raw_output", "")

    patterns: Dict[str, List[str]] = {
        "syntax_errors": [],
        "type_errors": [],
        "import_errors": [],
        "assertion_errors": [],
        "runtime_errors": [],
        "other": [],
    }

    for failure in failures:
        error_msg = failure.get("error", "") if isinstance(failure, dict) else str(failure)
        error_msg_lower = error_msg.lower()

        if "syntaxerror" in error_msg_lower:
            patterns["syntax_errors"].append(error_msg)
        elif "typeerror" in error_msg_lower:
            patterns["type_errors"].append(error_msg)
        elif "importerror" in error_msg_lower or "modulenotfounderror" in error_msg_lower:
            patterns["import_errors"].append(error_msg)
        elif "assertionerror" in error_msg_lower:
            patterns["assertion_errors"].append(error_msg)
        elif any(
            err in error_msg_lower
            for err in ["runtimeerror", "valueerror", "keyerror", "indexerror"]
        ):
            patterns["runtime_errors"].append(error_msg)
        else:
            patterns["other"].append(error_msg)

    # Also check raw error output
    if error_output:
        error_lower = error_output.lower()
        if "syntaxerror" in error_lower and not patterns["syntax_errors"]:
            patterns["syntax_errors"].append(error_output[:500])

    dominant = (
        max(patterns.keys(), key=lambda k: len(patterns[k])) if any(patterns.values()) else "none"
    )

    return {
        "error_patterns": patterns,
        "dominant_error_type": dominant,
        "failures": failures,
    }


def format_solution_output(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Format solution output for benchmark submission.

    Args:
        ctx: Workflow context with solution data

    Returns:
        Dict with formatted solution for submission
    """
    solution_code = ctx.get("solution_code", "")
    modified_files = ctx.get("modified_files", [])
    test_results = ctx.get("test_results", {})
    task_id = ctx.get("task_id", ctx.get("problem_id", "unknown"))
    implementation_result = ctx.get("implementation_result", {})

    # Get code from implementation result if not directly available
    if not solution_code and isinstance(implementation_result, dict):
        solution_code = implementation_result.get("code", "")
        modified_files = implementation_result.get("files", modified_files)

    return {
        "task_id": task_id,
        "solution": {
            "code": solution_code,
            "files": modified_files,
        },
        "metadata": {
            "test_pass_rate": test_results.get("pass_rate", 0),
            "iterations": ctx.get("fix_iterations", 1),
            "passed": test_results.get("passed", 0),
            "failed": test_results.get("failed", 0),
        },
        "status": "completed",
    }


# =============================================================================
# Registry Exports
# =============================================================================

# Conditions available in YAML workflows
CONDITIONS = {
    "solution_quality_check": solution_quality_check,
    "test_coverage_check": test_coverage_check,
    "complexity_check": complexity_check,
    # Additional conditions for workflow nodes
    "test_execution_status": test_execution_status,
    "should_continue_fixing": should_continue_fixing,
    "code_complexity_check": code_complexity_check,
    "verification_status": verification_status,
    "escalation_decision": escalation_decision,
    "passk_progress_check": passk_progress_check,
}

# Transforms available in YAML workflows
TRANSFORMS = {
    "extract_patch": extract_patch,
    "merge_tool_results": merge_tool_results,
    "format_for_evaluation": format_for_evaluation,
    # Additional transforms for workflow nodes
    "extract_error_patterns": extract_error_patterns,
    "format_solution_output": format_solution_output,
}

# Handlers available for compute nodes
HANDLERS = {
    "run_tests": RunTestsHandler(),
    "validate_patch": ValidatePatchHandler(),
}


def register_handlers() -> None:
    """Register Benchmark handlers with the workflow executor."""
    from victor.workflows.compute_registry import register_compute_handler

    for name, handler in HANDLERS.items():
        register_compute_handler(name, handler)
        logger.debug(f"Registered Benchmark handler: {name}")


__all__ = [
    # Enums
    "SolutionQuality",
    "CoverageStatus",
    "ComplexityLevel",
    # Conditions
    "solution_quality_check",
    "test_coverage_check",
    "complexity_check",
    "test_execution_status",
    "should_continue_fixing",
    "code_complexity_check",
    "verification_status",
    "escalation_decision",
    "passk_progress_check",
    # Transforms
    "extract_patch",
    "merge_tool_results",
    "format_for_evaluation",
    "extract_error_patterns",
    "format_solution_output",
    # Handlers
    "RunTestsHandler",
    "ValidatePatchHandler",
    "HANDLERS",
    "register_handlers",
    # Registries
    "CONDITIONS",
    "TRANSFORMS",
]
