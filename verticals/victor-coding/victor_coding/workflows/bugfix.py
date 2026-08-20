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

"""Bug fix workflow for coding vertical.

Provides a multi-step workflow for investigating and fixing bugs
with systematic debugging and verification.
"""

from typing import Any, Dict

from victor_contracts.workflow_runtime import (
    WorkflowBuilder,
    WorkflowDefinition,
    workflow,
)


def _check_verification_result(ctx: Dict[str, Any]) -> str:
    """Check if tests pass after fix.

    Args:
        ctx: Workflow context with verification_result

    Returns:
        'pass' if tests pass, 'fail' otherwise
    """
    result = ctx.get("verification_result", {})
    if isinstance(result, dict):
        tests_pass = result.get("tests_pass", True)
        if not tests_pass:
            return "fail"
    # Also check for explicit test failures in string result
    if isinstance(result, str):
        if "FAILED" in result or "ERROR" in result:
            return "fail"
    return "pass"


@workflow("bug_fix", "Systematic bug investigation and fix")
def bug_fix_workflow() -> WorkflowDefinition:
    """Create the bug fix workflow.

    This workflow guides agents through:
    1. Investigate - Analyze bug root cause
    2. Diagnose - Identify exact cause and plan fix
    3. Fix - Apply the fix
    4. Verify - Run tests to confirm fix
    5. Commit - Commit the fix

    Returns:
        WorkflowDefinition for bug fix
    """
    return (
        WorkflowBuilder("bug_fix")
        .set_metadata("category", "coding")
        .set_metadata("complexity", "medium")
        # Step 1: Investigate bug
        .add_agent(
            "investigate",
            role="researcher",
            goal="Investigate bug root cause through code analysis and logs",
            tool_budget=25,
            allowed_tools=[
                "read_file",
                "grep",
                "code_search",
                "semantic_code_search",
                "bash",
                "git_log",
                "references",
                "symbols",
            ],
            output_key="investigation",
        )
        # Step 2: Diagnose and plan fix
        .add_agent(
            "diagnose",
            role="planner",
            goal="Diagnose the exact issue and plan the fix",
            tool_budget=10,
            allowed_tools=[
                "read_file",
                "grep",
            ],
            input_mapping={"findings": "investigation"},
            output_key="fix_plan",
        )
        # Step 3: Apply fix
        .add_agent(
            "fix",
            role="executor",
            goal="Apply the fix according to diagnosis",
            tool_budget=20,
            allowed_tools=[
                "read_file",
                "edit_files",
                "write_file",
                "bash",
            ],
            input_mapping={"plan": "fix_plan"},
            output_key="fix_result",
        )
        # Step 4: Verify fix
        .add_agent(
            "verify",
            role="reviewer",
            goal="Run tests to verify the fix works",
            tool_budget=15,
            allowed_tools=[
                "bash",
                "run_tests",
                "test_file",
                "git_diff",
            ],
            output_key="verification_result",
        )
        # Condition: pass or retry
        .add_condition(
            "check_verify",
            condition=_check_verification_result,
            branches={"pass": "commit", "fail": "fix"},
        )
        # Step 5: Commit
        .add_agent(
            "commit",
            role="executor",
            goal="Commit the bug fix with descriptive message",
            tool_budget=5,
            allowed_tools=["git_status", "git_diff", "bash"],
            next_nodes=[],  # Terminal
        )
        .build()
    )


@workflow("quick_fix", "Quick bug fix for simple issues")
def quick_fix_workflow() -> WorkflowDefinition:
    """Create a quick fix workflow for simple bugs.

    Faster workflow for obvious bugs that don't need deep investigation.

    Returns:
        WorkflowDefinition for quick fix
    """
    return (
        WorkflowBuilder("quick_fix")
        .set_metadata("category", "coding")
        .set_metadata("complexity", "low")
        # Quick investigation
        .add_agent(
            "investigate",
            role="researcher",
            goal="Quickly locate the bug in the code",
            tool_budget=10,
            allowed_tools=[
                "read_file",
                "grep",
                "code_search",
            ],
            output_key="bug_location",
        )
        # Fix directly
        .add_agent(
            "fix",
            role="executor",
            goal="Apply the fix",
            tool_budget=15,
            allowed_tools=[
                "read_file",
                "edit_files",
                "bash",
            ],
            input_mapping={"location": "bug_location"},
        )
        # Verify
        .add_agent(
            "verify",
            role="reviewer",
            goal="Run tests to confirm fix",
            tool_budget=10,
            allowed_tools=["bash", "run_tests"],
            next_nodes=[],
        )
        .build()
    )


__all__ = [
    "bug_fix_workflow",
    "quick_fix_workflow",
]
