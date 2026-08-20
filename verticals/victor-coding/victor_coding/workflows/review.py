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

"""Code review workflow for coding vertical.

Provides comprehensive code review workflow with parallel analysis
for security, style, and logic correctness.
"""

from typing import Any, Dict

from victor_contracts.workflow_runtime import (
    WorkflowBuilder,
    WorkflowDefinition,
    workflow,
)


def _transform_merge_reviews(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Merge parallel review results into a single context.

    Args:
        ctx: Context with individual review results

    Returns:
        Updated context with merged reviews
    """
    reviews = {
        "security": ctx.get("security_review", {}),
        "style": ctx.get("style_review", {}),
        "logic": ctx.get("logic_review", {}),
    }
    ctx["merged_reviews"] = reviews
    return ctx


@workflow("code_review", "Comprehensive code review with parallel analysis")
def code_review_workflow() -> WorkflowDefinition:
    """Create the comprehensive code review workflow.

    This workflow runs three review types in parallel:
    1. Security Review - Check for vulnerabilities
    2. Style Review - Check code style and conventions
    3. Logic Review - Check logic correctness

    Then synthesizes findings into a report.

    Returns:
        WorkflowDefinition for code review
    """
    return (
        WorkflowBuilder("code_review")
        .set_metadata("category", "coding")
        .set_metadata("complexity", "medium")
        # Initial analysis to identify files to review
        .add_agent(
            "identify",
            role="researcher",
            goal="Identify files and changes to review",
            tool_budget=10,
            allowed_tools=[
                "git_diff",
                "git_status",
                "list_directory",
                "read_file",
            ],
            output_key="files_to_review",
        )
        # Parallel security review
        .add_agent(
            "security",
            role="researcher",
            name="Security Reviewer",
            goal="Check for security vulnerabilities and issues",
            tool_budget=15,
            allowed_tools=[
                "read_file",
                "grep",
                "code_search",
            ],
            input_mapping={"files": "files_to_review"},
            output_key="security_review",
            next_nodes=["synthesize"],  # Jump to synthesize after parallel
        )
        # Parallel style review
        .add_agent(
            "style",
            role="researcher",
            name="Style Reviewer",
            goal="Check code style, formatting, and conventions",
            tool_budget=15,
            allowed_tools=[
                "read_file",
                "bash",  # For linters
            ],
            input_mapping={"files": "files_to_review"},
            output_key="style_review",
            next_nodes=["synthesize"],
        )
        # Parallel logic review
        .add_agent(
            "logic",
            role="researcher",
            name="Logic Reviewer",
            goal="Check logic correctness and edge cases",
            tool_budget=15,
            allowed_tools=[
                "read_file",
                "grep",
                "references",
                "symbols",
            ],
            input_mapping={"files": "files_to_review"},
            output_key="logic_review",
            next_nodes=["synthesize"],
        )
        # Make parallel execution explicit
        .add_parallel(
            "parallel_reviews",
            parallel_nodes=["security", "style", "logic"],
            join_strategy="all",
            next_nodes=["synthesize"],
        )
        # Chain identify to parallel block
        .chain("identify", "parallel_reviews")
        # Synthesize all reviews
        .add_agent(
            "synthesize",
            role="planner",
            name="Review Synthesizer",
            goal="Synthesize all review findings into a comprehensive report",
            tool_budget=10,
            allowed_tools=["read_file"],
            input_mapping={
                "security": "security_review",
                "style": "style_review",
                "logic": "logic_review",
            },
            output_key="review_report",
            next_nodes=[],  # Terminal
        )
        .build()
    )


@workflow("quick_review", "Quick code review for small changes")
def quick_review_workflow() -> WorkflowDefinition:
    """Create a quick review workflow for small changes.

    Single-pass review for minor changes.

    Returns:
        WorkflowDefinition for quick review
    """
    return (
        WorkflowBuilder("quick_review")
        .set_metadata("category", "coding")
        .set_metadata("complexity", "low")
        # Get changes
        .add_agent(
            "identify",
            role="researcher",
            goal="Identify changes to review",
            tool_budget=5,
            allowed_tools=["git_diff", "git_status"],
            output_key="changes",
        )
        # Single review pass
        .add_agent(
            "review",
            role="reviewer",
            goal="Review changes for issues",
            tool_budget=15,
            allowed_tools=[
                "read_file",
                "grep",
                "bash",
            ],
            input_mapping={"changes": "changes"},
            output_key="review_result",
            next_nodes=[],
        )
        .build()
    )


@workflow("pr_review", "Pull request review workflow")
def pr_review_workflow() -> WorkflowDefinition:
    """Create a PR review workflow.

    Reviews a pull request with focus on:
    - Change summary
    - Impact analysis
    - Test coverage
    - Actionable feedback

    Returns:
        WorkflowDefinition for PR review
    """
    return (
        WorkflowBuilder("pr_review")
        .set_metadata("category", "coding")
        .set_metadata("complexity", "medium")
        # Fetch PR changes
        .add_agent(
            "fetch",
            role="researcher",
            goal="Fetch and analyze PR changes",
            tool_budget=10,
            allowed_tools=[
                "bash",  # gh cli
                "git_diff",
                "git_log",
            ],
            output_key="pr_changes",
        )
        # Analyze impact
        .add_agent(
            "analyze",
            role="researcher",
            goal="Analyze impact of changes on the codebase",
            tool_budget=20,
            allowed_tools=[
                "read_file",
                "grep",
                "references",
                "symbols",
                "code_search",
            ],
            input_mapping={"changes": "pr_changes"},
            output_key="impact_analysis",
        )
        # Check tests
        .add_agent(
            "test_check",
            role="reviewer",
            goal="Check test coverage and run tests",
            tool_budget=15,
            allowed_tools=[
                "bash",
                "run_tests",
                "read_file",
            ],
            output_key="test_results",
        )
        # Generate review
        .add_agent(
            "generate_review",
            role="planner",
            goal="Generate actionable review comments",
            tool_budget=10,
            allowed_tools=["read_file"],
            input_mapping={
                "changes": "pr_changes",
                "impact": "impact_analysis",
                "tests": "test_results",
            },
            output_key="review_comments",
            next_nodes=[],
        )
        .build()
    )


__all__ = [
    "code_review_workflow",
    "quick_review_workflow",
    "pr_review_workflow",
]
