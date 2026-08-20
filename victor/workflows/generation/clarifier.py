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

"""Ambiguity detection and resolution for workflow requirements.

This module provides interactive and automated ambiguity resolution
for extracted workflow requirements.

Design Principles (SOLID):
    - SRP: Separate detection, resolution, and question generation
    - OCP: Extensible resolution strategies
    - LSP: All resolvers implement the same interface
    - ISP: Focused interfaces for each resolution type
    - DIP: Depends on WorkflowRequirements abstraction

Key Features:
    - Ambiguity Detection: Identify missing/unclear requirements
    - Interactive Resolution: Ask users for clarification
    - Assumption-Based Resolution: Make reasonable guesses
    - Priority-Based Ordering: Handle critical ambiguities first

Example:
    from victor.workflows.generation.clarifier import (
        AmbiguityDetector,
        AmbiguityResolver,
        InteractiveClarifier,
    )

    detector = AmbiguityDetector()
    ambiguities = detector.detect(requirements)

    resolver = AmbiguityResolver(orchestrator)
    resolved = await resolver.resolve(
        requirements,
        ambiguities,
        strategy="interactive"
    )
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from victor.workflows.generation.requirements import (
    Ambiguity,
    ContextRequirements,
    FunctionalRequirements,
    QualityRequirements,
    StructuralRequirements,
    WorkflowRequirements,
)

if TYPE_CHECKING:
    from victor.framework.protocols import (
        FrameworkOrchestratorProtocol as OrchestratorProtocol,
    )

logger = logging.getLogger(__name__)


# =============================================================================
# Question Types
# =============================================================================


@dataclass
class Question:
    """A question to ask the user.

    Attributes:
        text: Question text
        options: Multiple choice options (None = free-form)
        default: Default answer
        field: Field this question resolves
        ambiguity_ref: Reference to the ambiguity this addresses

    Example:
        question = Question(
            text="What role should perform this task?",
            options=["researcher", "executor", "planner"],
            default="executor",
            field="functional.tasks.task_1.role"
        )
    """

    text: str
    options: Optional[List[str]] = None
    default: Optional[str] = None
    field: str = ""
    ambiguity_ref: Optional[Ambiguity] = None


class QuestionGenerator:
    """Generate clarifying questions from ambiguities.

    Maps ambiguity types to appropriate question templates.
    """

    # Question templates for each ambiguity type
    TEMPLATES = {
        "missing_tasks": Question(
            text="What should this workflow do?",
            default="Not specified",
        ),
        "missing_role": Question(
            text="What role should perform this task?",
            options=["researcher", "executor", "planner", "reviewer", "writer"],
            default="executor",
        ),
        "missing_condition": Question(
            text="What condition should trigger this branch?",
            default="success",
        ),
        "vague_description": Question(
            text="Can you be more specific about what this task does?",
            default="Use current description",
        ),
        "infeasible_constraint": Question(
            text="{message}",
            options=["Adjust constraint", "Reduce tasks", "Ignore warning"],
            default="Adjust constraint",
        ),
        "missing_success_criteria": Question(
            text="How do we know the workflow succeeded?",
            default="Task completed successfully",
        ),
        "circular_dependency": Question(
            text="How should we break the circular dependency?",
            options=["Remove last dependency", "Merge tasks", "Ignore warning"],
            default="Remove last dependency",
        ),
    }

    def generate(self, ambiguity: Ambiguity) -> Question:
        """Generate a question for an ambiguity.

        Args:
            ambiguity: The ambiguity to clarify

        Returns:
            Question to ask user
        """
        template = self.TEMPLATES.get(ambiguity.type)

        if template:
            # Customize template with ambiguity details
            text = template.text
            if "{message}" in text:
                text = text.format(message=ambiguity.message)
            if "{task_id}" in text:
                text = text.format(task_id=self._extract_task_id(ambiguity.field))

            return Question(
                text=text,
                options=ambiguity.options or template.options,
                default=template.default,
                field=ambiguity.field,
                ambiguity_ref=ambiguity,
            )
        else:
            # Generic question
            return Question(
                text=f"{ambiguity.message}. {ambiguity.suggestion}",
                options=ambiguity.options,
                field=ambiguity.field,
                ambiguity_ref=ambiguity,
            )

    def _extract_task_id(self, field: str) -> str:
        """Extract task ID from field path."""
        # Extract from "functional.tasks.task_1.role"
        match = re.search(r"tasks\.([^.]+)\.", field)
        return match.group(1) if match else "unknown"


# =============================================================================
# Ambiguity Detector
# =============================================================================


class AmbiguityDetector:
    """Detect ambiguities in extracted requirements.

    Finds:
    - Missing required fields
    - Contradictory requirements
    - Vague descriptions
    - Infeasible combinations

    Attributes:
        _vague_words: Words that indicate vague descriptions
        _min_description_length: Minimum characters for a good description

    Example:
        detector = AmbiguityDetector()
        ambiguities = detector.detect(requirements)
        # Returns list of Ambiguity objects sorted by severity
    """

    def __init__(self, min_description_length: int = 15):
        """Initialize ambiguity detector.

        Args:
            min_description_length: Minimum characters for non-vague descriptions
        """
        self._vague_words = [
            "something",
            "stuff",
            "things",
            "handle",
            "process",
            "whatever",
        ]
        self._min_description_length = min_description_length

    def detect(self, requirements: WorkflowRequirements) -> List[Ambiguity]:
        """Detect all ambiguities in requirements.

        Args:
            requirements: Requirements to check

        Returns:
            List of Ambiguity objects sorted by severity
        """
        ambiguities = []

        # Check for missing functional requirements
        ambiguities.extend(self._check_functional_gaps(requirements))

        # Check for structural inconsistencies
        ambiguities.extend(self._check_structural_conflicts(requirements))

        # Check for vague descriptions
        ambiguities.extend(self._check_vagueness(requirements))

        # Check for infeasible constraints
        ambiguities.extend(self._check_feasibility(requirements))

        # Sort by severity (highest first)
        ambiguities.sort(key=lambda a: a.severity, reverse=True)

        return ambiguities

    def _check_functional_gaps(self, requirements: WorkflowRequirements) -> List[Ambiguity]:
        """Check for missing functional requirements."""
        gaps = []

        # Check: At least one task
        if not requirements.functional.tasks:
            gaps.append(
                Ambiguity(
                    type="missing_tasks",
                    severity=10,
                    message="No tasks detected",
                    suggestion="What should this workflow do?",
                    field="functional.tasks",
                )
            )

        # Check: Tasks have roles (for agent tasks)
        for task in requirements.functional.tasks:
            if task.task_type == "agent" and not task.role:
                gaps.append(
                    Ambiguity(
                        type="missing_role",
                        severity=7,
                        message=f"Task '{task.id}' has no agent role specified",
                        suggestion="What role should perform this task?",
                        field=f"functional.tasks.{task.id}.role",
                        options=[
                            "researcher",
                            "executor",
                            "planner",
                            "reviewer",
                            "writer",
                        ],
                    )
                )

        # Check: Success criteria defined
        if not requirements.functional.success_criteria:
            gaps.append(
                Ambiguity(
                    type="missing_success_criteria",
                    severity=5,
                    message="No success criteria defined",
                    suggestion="How do we know the workflow succeeded?",
                    field="functional.success_criteria",
                )
            )

        return gaps

    def _check_structural_conflicts(self, requirements: WorkflowRequirements) -> List[Ambiguity]:
        """Check for structural inconsistencies."""
        conflicts = []

        # Check: Circular dependencies
        graph = self._build_dependency_graph(requirements)
        cycles = self._detect_cycles(graph)
        if cycles:
            cycle_path = " → ".join(cycles[0])
            conflicts.append(
                Ambiguity(
                    type="circular_dependency",
                    severity=9,
                    message=f"Circular dependency detected: {cycle_path}",
                    suggestion="Break the cycle by removing one dependency",
                    field="structural.dependencies",
                )
            )

        # Check: Parallel tasks with sequential dependencies
        if (
            requirements.structural.execution_order == "parallel"
            and requirements.structural.dependencies
        ):
            conflicts.append(
                Ambiguity(
                    type="contradictory_structure",
                    severity=6,
                    message="Parallel execution with dependencies may be inefficient",
                    suggestion="Consider sequential execution or remove dependencies",
                    field="structural.execution_order",
                )
            )

        # Check: Branch without condition
        for branch in requirements.structural.branches:
            if not branch.condition or branch.condition == "unknown":
                conflicts.append(
                    Ambiguity(
                        type="missing_condition",
                        severity=8,
                        message=f"Branch '{branch.condition_id}' has no condition",
                        suggestion="What condition should trigger this branch?",
                        field=f"structural.branches.{branch.condition_id}.condition",
                    )
                )

        return conflicts

    def _check_vagueness(self, requirements: WorkflowRequirements) -> List[Ambiguity]:
        """Check for vague descriptions."""
        vague = []

        # Vague task descriptions
        for task in requirements.functional.tasks:
            words = task.description.lower().split()
            if any(v in words for v in self._vague_words):
                vague.append(
                    Ambiguity(
                        type="vague_description",
                        severity=4,
                        message=f"Task '{task.id}' has vague description",
                        suggestion="Be more specific about what this task does",
                        field=f"functional.tasks.{task.id}.description",
                    )
                )

        # Vague success criteria
        for criteria in requirements.functional.success_criteria:
            if len(criteria) < 10:
                vague.append(
                    Ambiguity(
                        type="vague_criteria",
                        severity=3,
                        message=f"Success criteria is too brief: '{criteria}'",
                        suggestion="Provide specific, measurable criteria",
                        field="functional.success_criteria",
                    )
                )

        return vague

    def _check_feasibility(self, requirements: WorkflowRequirements) -> List[Ambiguity]:
        """Check for infeasible constraints."""
        infeasible = []

        # Check: Too many tool calls for budget
        if requirements.quality.max_tool_calls:
            estimated_calls = len(requirements.functional.tasks) * 10  # Rough estimate
            if estimated_calls > requirements.quality.max_tool_calls:
                infeasible.append(
                    Ambiguity(
                        type="infeasible_constraint",
                        severity=7,
                        message=f"Estimated {estimated_calls} tool calls exceeds budget of {requirements.quality.max_tool_calls}",
                        suggestion=f"Increase tool budget to {estimated_calls} or reduce tasks",
                        field="quality.max_tool_calls",
                    )
                )

        # Check: Timeout too short for task count
        if requirements.quality.max_duration_seconds:
            min_seconds = len(requirements.functional.tasks) * 30  # 30s per task min
            if requirements.quality.max_duration_seconds < min_seconds:
                infeasible.append(
                    Ambiguity(
                        type="infeasible_constraint",
                        severity=8,
                        message=f"Timeout {requirements.quality.max_duration_seconds}s too short for {len(requirements.functional.tasks)} tasks",
                        suggestion=f"Increase timeout to at least {min_seconds}s",
                        field="quality.max_duration_seconds",
                    )
                )

        return infeasible

    def _build_dependency_graph(self, requirements: WorkflowRequirements) -> Dict[str, List[str]]:
        """Build dependency graph from requirements."""
        graph = {}
        for task_id, deps in requirements.structural.dependencies.items():
            graph[task_id] = deps
        return graph

    def _detect_cycles(self, graph: Dict[str, List[str]]) -> List[List[str]]:
        """Detect cycles in dependency graph using DFS.

        Args:
            graph: Adjacency list representation

        Returns:
            List of cycles (each cycle is a list of node IDs)
        """
        cycles = []
        visited = set()
        rec_stack = set()
        path = []

        def dfs(node: str):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)
                    return True

            path.pop()
            rec_stack.remove(node)
            return False

        for node in graph:
            if node not in visited:
                dfs(node)

        return cycles


# =============================================================================
# Ambiguity Resolver
# =============================================================================


class AmbiguityResolver:
    """Resolve ambiguities in workflow requirements.

    Strategies:
    1. Interactive: Ask user for clarification (preferred)
    2. Assumptions: Make reasonable assumptions with confidence scores
    3. Defaults: Use template defaults for common patterns

    Attributes:
        _orchestrator: Orchestrator for LLM access
        _question_generator: Generate questions from ambiguities

    Example:
        resolver = AmbiguityResolver(orchestrator)
        resolved = await resolver.resolve(
            requirements,
            ambiguities,
            strategy="interactive"
        )
    """

    def __init__(self, orchestrator: OrchestratorProtocol):
        """Initialize ambiguity resolver.

        Args:
            orchestrator: Orchestrator for LLM access
        """
        self._orchestrator = orchestrator
        self._question_generator = QuestionGenerator()

    async def resolve(
        self,
        requirements: WorkflowRequirements,
        ambiguities: List[Ambiguity],
        strategy: str = "interactive",
    ) -> WorkflowRequirements:
        """Resolve ambiguities using specified strategy.

        Args:
            requirements: Requirements with ambiguities
            ambiguities: List of detected ambiguities
            strategy: Resolution strategy (interactive, assumptions, defaults)

        Returns:
            Resolved requirements

        Raises:
            ValueError: If unknown strategy
        """
        if not ambiguities:
            return requirements

        logger.info(f"Resolving {len(ambiguities)} ambiguities using {strategy} strategy")

        if strategy == "interactive":
            return await self._resolve_interactively(requirements, ambiguities)
        elif strategy == "assumptions":
            return await self._resolve_with_assumptions(requirements, ambiguities)
        elif strategy == "defaults":
            return self._resolve_with_defaults(requirements, ambiguities)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    async def _resolve_interactively(
        self,
        requirements: WorkflowRequirements,
        ambiguities: List[Ambiguity],
    ) -> WorkflowRequirements:
        """Resolve by asking user questions.

        Note: This is a placeholder implementation. The actual interactive
        resolution should use Rich prompts and handle user input properly.

        Args:
            requirements: Requirements with ambiguities
            ambiguities: List of ambiguities

        Returns:
            Resolved requirements
        """
        # For now, just use assumptions
        # In production, this would use Rich.prompt.Prompt
        logger.warning("Interactive resolution not fully implemented, using assumptions")
        return await self._resolve_with_assumptions(requirements, ambiguities)

    async def _resolve_with_assumptions(
        self,
        requirements: WorkflowRequirements,
        ambiguities: List[Ambiguity],
    ) -> WorkflowRequirements:
        """Resolve by making assumptions.

        Args:
            requirements: Requirements with ambiguities
            ambiguities: List of ambiguities

        Returns:
            Resolved requirements
        """
        # Apply default resolutions based on ambiguity type
        for ambiguity in ambiguities:
            requirements = self._apply_default_resolution(requirements, ambiguity)

        return requirements

    def _resolve_with_defaults(
        self,
        requirements: WorkflowRequirements,
        ambiguities: List[Ambiguity],
    ) -> WorkflowRequirements:
        """Resolve using template defaults.

        Args:
            requirements: Requirements with ambiguities
            ambiguities: List of ambiguities

        Returns:
            Resolved requirements
        """
        # Similar to assumptions but more conservative
        for ambiguity in ambiguities:
            requirements = self._apply_default_resolution(requirements, ambiguity)

        return requirements

    def _apply_default_resolution(
        self,
        requirements: WorkflowRequirements,
        ambiguity: Ambiguity,
    ) -> WorkflowRequirements:
        """Apply a default resolution for an ambiguity.

        Args:
            requirements: Requirements to modify
            ambiguity: Ambiguity to resolve

        Returns:
            Modified requirements
        """
        # Parse field path
        parts = ambiguity.field.split(".")

        if len(parts) < 2:
            return requirements

        category = parts[0]

        if category == "functional":
            return self._apply_functional_resolution(requirements, parts[1:], ambiguity)
        elif category == "structural":
            return self._apply_structural_resolution(requirements, parts[1:], ambiguity)
        elif category == "quality":
            return self._apply_quality_resolution(requirements, parts[1:], ambiguity)
        elif category == "context":
            return self._apply_context_resolution(requirements, parts[1:], ambiguity)

        return requirements

    def _apply_functional_resolution(
        self,
        requirements: WorkflowRequirements,
        parts: List[str],
        ambiguity: Ambiguity,
    ) -> WorkflowRequirements:
        """Apply resolution to functional requirements."""
        if len(parts) >= 3 and parts[0] == "tasks":
            # Extract task ID and field
            task_id = parts[1]
            field = parts[2] if len(parts) > 2 else None

            # Find the task
            for task in requirements.functional.tasks:
                if task.id == task_id:
                    if field == "role":
                        # Default to executor
                        if ambiguity.options:
                            task.role = ambiguity.options[0]
                        else:
                            task.role = "executor"
                    elif field == "description":
                        # Keep current or use suggestion
                        pass  # Keep current description

        return requirements

    def _apply_structural_resolution(
        self,
        requirements: WorkflowRequirements,
        parts: List[str],
        ambiguity: Ambiguity,
    ) -> WorkflowRequirements:
        """Apply resolution to structural requirements."""
        # Handle structural resolutions
        if "dependencies" in parts and ambiguity.type == "circular_dependency":
            # Remove last dependency to break cycle
            # This is simplified - real implementation would be smarter
            pass

        return requirements

    def _apply_quality_resolution(
        self,
        requirements: WorkflowRequirements,
        parts: List[str],
        ambiguity: Ambiguity,
    ) -> WorkflowRequirements:
        """Apply resolution to quality requirements."""
        # Handle quality constraint adjustments
        if "max_duration_seconds" in parts:
            # Increase timeout
            requirements.quality.max_duration_seconds = (
                requirements.quality.max_duration_seconds or 300
            ) + 300

        if "max_tool_calls" in parts:
            # Increase budget
            requirements.quality.max_tool_calls = (requirements.quality.max_tool_calls or 100) + 50

        return requirements

    def _apply_context_resolution(
        self,
        requirements: WorkflowRequirements,
        parts: List[str],
        ambiguity: Ambiguity,
    ) -> WorkflowRequirements:
        """Apply resolution to context requirements."""
        # Handle context resolutions
        if "vertical" in parts and ambiguity.options:
            requirements.context.vertical = ambiguity.options[0]

        return requirements


# =============================================================================
# Interactive Clarifier (Rich-based)
# =============================================================================


class InteractiveClarifier:
    """Interactive clarification system for workflow requirements.

    Engages user in conversation to resolve ambiguities.

    Note: This requires Rich for interactive prompts.
    Falls back to assumption-based resolution if Rich unavailable.

    Attributes:
        _orchestrator: Orchestrator for LLM access
        _detector: Ambiguity detector
        _resolver: Ambiguity resolver

    Example:
        clarifier = InteractiveClarifier(orchestrator)
        resolved = await clarifier.clarify(requirements)
        # Asks user questions interactively
    """

    def __init__(self, orchestrator: OrchestratorProtocol):
        """Initialize interactive clarifier.

        Args:
            orchestrator: Orchestrator for LLM access
        """
        self._orchestrator = orchestrator
        self._detector = AmbiguityDetector()
        self._resolver = AmbiguityResolver(orchestrator)

    async def clarify(
        self,
        requirements: WorkflowRequirements,
    ) -> WorkflowRequirements:
        """Clarify requirements through interactive dialogue.

        Args:
            requirements: Initial requirements

        Returns:
            Refined requirements with ambiguities resolved
        """
        # Detect ambiguities
        ambiguities = self._detector.detect(requirements)

        if not ambiguities:
            logger.info("No ambiguities detected")
            return requirements

        logger.info(f"Found {len(ambiguities)} ambiguities")

        # Try interactive, fall back to assumptions
        try:
            return await self._resolver.resolve(
                requirements,
                ambiguities,
                strategy="interactive",
            )
        except Exception as e:
            logger.warning(f"Interactive resolution failed: {e}, using assumptions")
            return await self._resolver.resolve(
                requirements,
                ambiguities,
                strategy="assumptions",
            )


__all__ = [
    "AmbiguityDetector",
    "AmbiguityResolver",
    "InteractiveClarifier",
    "Question",
    "QuestionGenerator",
]
