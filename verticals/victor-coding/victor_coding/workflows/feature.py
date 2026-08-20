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

"""Feature implementation workflow for coding vertical.

Provides a multi-step workflow for implementing new features with
research, planning, implementation, and review stages.
"""

from typing import Any, Dict

from victor_contracts.workflow_runtime import (
    WorkflowBuilder,
    WorkflowDefinition,
    workflow,
)


def _check_review_result(ctx: Dict[str, Any]) -> str:
    """Check review result and decide next step.

    Args:
        ctx: Workflow context with review_result

    Returns:
        'fix' if issues found, 'done' otherwise
    """
    review_result = ctx.get("review_result", {})
    if isinstance(review_result, dict):
        has_issues = review_result.get("issues") or review_result.get("has_issues")
        if has_issues:
            return "fix"
    return "done"


@workflow("feature_implementation", "End-to-end feature development with review")
def feature_implementation_workflow() -> WorkflowDefinition:
    """Create the feature implementation workflow.

    This workflow guides agents through:
    1. Research - Analyze codebase for relevant patterns
    2. Plan - Create implementation plan
    3. Implement - Write the code
    4. Review - Review and test
    5. Finalize - Commit changes

    Returns:
        WorkflowDefinition for feature implementation
    """
    return (
        WorkflowBuilder("feature_implementation")
        .set_metadata("category", "coding")
        .set_metadata("complexity", "high")
        # Step 1: Research existing patterns
        .add_agent(
            "research",
            role="researcher",
            goal="Analyze codebase for relevant patterns and dependencies",
            tool_budget=20,
            allowed_tools=[
                "read_file",
                "grep",
                "code_search",
                "semantic_code_search",
                "project_overview",
                "symbols",
                "references",
                "list_directory",
            ],
            output_key="research_findings",
        )
        # Step 2: Plan implementation
        .add_agent(
            "plan",
            role="planner",
            goal="Create detailed implementation plan based on research",
            tool_budget=10,
            allowed_tools=[
                "read_file",
                "project_overview",
                "plan_files",
            ],
            input_mapping={"findings": "research_findings"},
            output_key="implementation_plan",
        )
        # Human approval gate before implementation
        .add_hitl_approval(
            "approve_plan",
            prompt="Review the implementation plan before proceeding:",
            context_keys=["implementation_plan"],
            timeout=300.0,
            fallback="continue",  # Continue if no response
        )
        # Step 3: Implement feature
        .add_agent(
            "implement",
            role="executor",
            goal="Implement the feature according to plan",
            tool_budget=40,
            allowed_tools=[
                "read_file",
                "write_file",
                "edit_files",
                "bash",
                "git_status",
                "git_diff",
            ],
            input_mapping={"plan": "implementation_plan"},
            output_key="implementation_result",
        )
        # Step 4: Review and test
        .add_agent(
            "review",
            role="reviewer",
            goal="Review implementation and run tests",
            tool_budget=20,
            allowed_tools=[
                "read_file",
                "bash",
                "run_tests",
                "git_diff",
                "test_file",
            ],
            input_mapping={"code": "implementation_result"},
            output_key="review_result",
        )
        # Condition: fix issues or complete
        .add_condition(
            "check_review",
            condition=_check_review_result,
            branches={"fix": "implement", "done": "finalize"},
        )
        # Step 5: Finalize
        .add_agent(
            "finalize",
            role="executor",
            goal="Commit changes and summarize work done",
            tool_budget=5,
            allowed_tools=["git_status", "git_diff", "bash"],
            next_nodes=[],  # Terminal node
        )
        .build()
    )


@workflow("quick_feature", "Fast feature implementation without review loop")
def quick_feature_workflow() -> WorkflowDefinition:
    """Create a quick feature workflow without review loop.

    For simple features where full review cycle is not needed.

    Returns:
        WorkflowDefinition for quick feature implementation
    """
    return (
        WorkflowBuilder("quick_feature")
        .set_metadata("category", "coding")
        .set_metadata("complexity", "low")
        # Research
        .add_agent(
            "research",
            role="researcher",
            goal="Quick scan of relevant code patterns",
            tool_budget=10,
            allowed_tools=[
                "read_file",
                "grep",
                "code_search",
                "list_directory",
            ],
            output_key="findings",
        )
        # Implement directly
        .add_agent(
            "implement",
            role="executor",
            goal="Implement the feature",
            tool_budget=25,
            allowed_tools=[
                "read_file",
                "write_file",
                "edit_files",
                "bash",
            ],
            input_mapping={"context": "findings"},
            output_key="result",
        )
        # Quick verify
        .add_agent(
            "verify",
            role="reviewer",
            goal="Run tests and verify implementation",
            tool_budget=10,
            allowed_tools=["bash", "run_tests"],
            next_nodes=[],
        )
        .build()
    )


__all__ = [
    "feature_implementation_workflow",
    "quick_feature_workflow",
]
