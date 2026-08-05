# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Generic escape hatches for YAML workflows.

Complex conditions and transforms that cannot be expressed in YAML.
These are registered with the YAML workflow loader for use in condition nodes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


# =============================================================================
# Condition Functions
# =============================================================================


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


# =============================================================================
# Transform Functions
# =============================================================================


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
    "complexity_assessment": complexity_assessment,
    "complexity_check": complexity_check,
}

# Transforms available in YAML workflows
TRANSFORMS = {
    "format_implementation_plan": format_implementation_plan,
}


def register_escape_hatches(registry: Any) -> None:
    """Register these generic conditions/transforms into the global namespace.

    Follows the ``victor.escape_hatches`` provider convention
    (``register_escape_hatches(registry)``). Registers into the *global* namespace
    (``vertical=None``) so the conditions are available to every YAML workflow, with
    ``replace=True`` so the call is idempotent (safe across registry resets and repeated
    provider configuration). Provider-specific escape hatches still take precedence —
    they are merged on top of the global ones by the YAML config loader.
    """
    for name, fn in CONDITIONS.items():
        registry.register_condition(name, fn, vertical=None, replace=True)
    for name, fn in TRANSFORMS.items():
        registry.register_transform(name, fn, vertical=None, replace=True)


def ensure_global_escape_hatches_registered() -> None:
    """Idempotently register the generic escape hatches into the global registry."""
    from victor.framework.escape_hatch_registry import get_escape_hatch_registry

    register_escape_hatches(get_escape_hatch_registry())


__all__ = [
    # Conditions
    "complexity_assessment",
    "complexity_check",
    # Transforms
    "format_implementation_plan",
    # Registries
    "CONDITIONS",
    "TRANSFORMS",
    # Registration
    "register_escape_hatches",
    "ensure_global_escape_hatches_registered",
]
