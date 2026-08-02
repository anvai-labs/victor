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

"""Autonomous planning module for goal-oriented execution.

This package provides infrastructure for Victor to accept high-level goals
and autonomously generate and execute structured plans.

Components:
- ExecutionPlan: A plan consisting of steps with dependencies
- PlanStep: Individual step in a plan
- AutonomousPlanner: Generates and executes plans
- ReadableTaskPlan: Token-efficient schema with readable keywords for LLM planning
- TaskPlannerContext: Session context management for plans

Example:
    from victor.agent.planning import AutonomousPlanner, ExecutionPlan
    from victor.agent.planning.readable_schema import ReadableTaskPlan

    planner = AutonomousPlanner(orchestrator)

    # Generate plan using readable schema (token-efficient with clear keywords)
    readable_plan = await generate_task_plan(provider, "Add user auth")

    # Convert to full execution plan
    plan = readable_plan.to_execution_plan()

    # Or generate plan directly
    plan = await planner.plan_for_goal("Implement user authentication")

    # Review plan
    print(plan.to_markdown())

    # Execute plan
    result = await planner.execute_plan(plan, auto_approve=True)
"""

from victor.agent.planning.autonomous import (
    AutonomousPlanner,
    PLANNING_SYSTEM_PROMPT,
)
from victor.agent.planning.base import (
    ExecutionPlan,
    PlanResult,
    PlanStep,
    StepResult,
    StepStatus,
    StepType,
)
from victor.agent.planning.constants import (
    COMPLEXITY_KEYWORDS,
    COMPLEXITY_TOOL_LIMITS,
    DEFAULT_MIN_KEYWORD_MATCHES,
    DEFAULT_MIN_PLANNING_COMPLEXITY,
    DEFAULT_MIN_STEPS_THRESHOLD,
    STEP_INDICATORS,
    STEP_TO_TASK_TYPE,
    STEP_TOOL_MAPPING,
)
from victor.agent.planning.readable_schema import (
    ReadableTaskPlan,
    TaskComplexity,
    TaskPlannerContext,
    generate_task_plan,
    plan_to_session_context,
    plan_to_workflow_yaml,
)
from victor.agent.planning.tool_selection import (
    StepAwareToolSelector,
)

# Export TaskPlan as primary alias for ReadableTaskPlan
TaskPlan = ReadableTaskPlan

__all__ = [
    # Base data structures
    "ExecutionPlan",
    "PlanResult",
    "PlanStep",
    "StepResult",
    "StepStatus",
    "StepType",
    # Readable schema (token-efficient LLM planning with clear keywords)
    "TaskPlan",  # Primary alias for ReadableTaskPlan
    "ReadableTaskPlan",
    "TaskComplexity",
    "TaskPlannerContext",
    "generate_task_plan",
    "plan_to_session_context",
    "plan_to_workflow_yaml",
    # Constants
    "COMPLEXITY_TOOL_LIMITS",
    "STEP_TOOL_MAPPING",
    "STEP_TO_TASK_TYPE",
    "COMPLEXITY_KEYWORDS",
    "STEP_INDICATORS",
    "DEFAULT_MIN_PLANNING_COMPLEXITY",
    "DEFAULT_MIN_STEPS_THRESHOLD",
    "DEFAULT_MIN_KEYWORD_MATCHES",
    # Context-aware tool selection
    "StepAwareToolSelector",
    # Planner
    "AutonomousPlanner",
    "PLANNING_SYSTEM_PROMPT",
]

# Legacy aliases for backward compatibility (will be deprecated in future)
CompactTaskPlan = ReadableTaskPlan
generate_compact_plan = generate_task_plan
