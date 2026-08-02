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

"""Workflow template library for common patterns.

This module provides pre-defined workflow templates that can be instantiated
with parameters from requirements. Templates serve as a fast fallback when
LLM generation fails or for simple, predictable patterns.

Design Principles (SOLID):
    - SRP: TemplateLibrary manages templates only (not generation)
    - OCP: Extensible via new templates without modifying library
    - LSP: All templates implement the same interface
    - ISP: Focused template matching and instantiation methods
    - DIP: Depends on WorkflowRequirements abstraction

Key Features:
    - Sequential workflow template
    - Conditional workflow template
    - Parallel workflow template
    - Research workflow template
    - Coding workflow template
    - DevOps workflow template
    - Template matching by keyword and structure

Example:
    from victor.workflows.generation.templates import TemplateLibrary

    library = TemplateLibrary()
    template = library.match_template(requirements, vertical="coding")
    schema = library.instantiate_template(template, requirements)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from victor.workflows.generation.requirements import WorkflowRequirements

logger = logging.getLogger(__name__)


# =============================================================================
# Template Types
# =============================================================================


class TemplateType(Enum):
    """Workflow template categories."""

    SEQUENTIAL = "sequential"  # Linear task sequence
    CONDITIONAL = "conditional"  # Branching based on conditions
    PARALLEL = "parallel"  # Concurrent execution
    LOOP = "loop"  # Repetitive with exit condition
    HIERARCHICAL = "hierarchical"  # Multi-level orchestration


# =============================================================================
# Workflow Template
# =============================================================================


@dataclass
class WorkflowTemplate:
    """A workflow template for common patterns.

    Attributes:
        name: Template identifier
        description: What this template does
        template_type: Type of workflow pattern
        verticals: Which verticals this applies to
        keywords: Keywords for matching (e.g., ["research", "analyze"])
        execution_order: Required execution order
        min_tasks: Minimum number of tasks
        max_tasks: Maximum number of tasks (None = unlimited)
        schema: Template schema with placeholders

    Example:
        template = WorkflowTemplate(
            name="sequential_research",
            description="Sequential research workflow",
            template_type=TemplateType.SEQUENTIAL,
            verticals=["research"],
            keywords=["research", "analyze", "investigate"],
            schema={...}
        )
    """

    name: str
    description: str
    template_type: TemplateType
    verticals: List[str]
    keywords: List[str]
    execution_order: str
    min_tasks: int = 1
    max_tasks: Optional[int] = None
    schema: Dict[str, Any] = field(default_factory=dict)

    def matches(
        self,
        requirements: WorkflowRequirements,
        vertical: str,
        keyword_match_threshold: float = 0.3,
    ) -> float:
        """Calculate match score for this template.

        Args:
            requirements: Workflow requirements to match
            vertical: Target vertical
            keyword_match_threshold: Minimum keyword score (0.0-1.0)

        Returns:
            Match score from 0.0 (no match) to 1.0 (perfect match).
            Returns 0.0 if vertical doesn't match (hard requirement).
        """
        # Vertical is a hard requirement - must match to get any score
        if vertical not in self.verticals:
            return 0.0

        score = 0.0

        # Check vertical (40% weight)
        score += 0.4

        # Check execution order (30% weight)
        if requirements.structural.execution_order == self.execution_order:
            score += 0.3

        # Check task count (20% weight)
        task_count = len(requirements.functional.tasks)
        if self.min_tasks <= task_count:
            if self.max_tasks is None or task_count <= self.max_tasks:
                score += 0.2

        # Check keyword match (10% weight)
        description_lower = requirements.description.lower()
        keyword_matches = sum(1 for kw in self.keywords if kw.lower() in description_lower)
        keyword_score = keyword_matches / max(len(self.keywords), 1)
        score += keyword_score * 0.1

        return min(score, 1.0)


# =============================================================================
# Template Library
# =============================================================================


class TemplateLibrary:
    """Library of workflow templates with matching and instantiation.

    Provides pre-defined templates for common workflow patterns across
    all verticals. Templates are parameterized and can be instantiated
    with specific requirements.

    Attributes:
        _templates: All registered templates

    Example:
        library = TemplateLibrary()

        # Find matching template
        template = library.match_template(requirements, vertical="research")

        # Instantiate with requirements
        schema = library.instantiate_template(template, requirements)

        # Use with StateGraph
        graph = StateGraph.from_schema(schema)
    """

    def __init__(self) -> None:
        """Initialize the template library."""
        self._templates = self._load_builtin_templates()
        logger.debug(f"Loaded {len(self._templates)} workflow templates")

    def match_template(
        self,
        requirements: WorkflowRequirements,
        vertical: str,
        min_score: float = 0.5,
    ) -> Optional[WorkflowTemplate]:
        """Find best matching template for requirements.

        Args:
            requirements: Workflow requirements
            vertical: Target vertical
            min_score: Minimum match score (0.0-1.0)

        Returns:
            Best matching template or None if no match meets threshold
        """
        best_template = None
        best_score = 0.0

        for template in self._templates:
            score = template.matches(requirements, vertical)
            if score > best_score:
                best_score = score
                best_template = template

        if best_score >= min_score:
            logger.debug(f"Template '{best_template.name}' matched with score {best_score:.2f}")
            return best_template

        logger.warning(f"No template matched minimum score {min_score}")
        return None

    def instantiate_template(
        self,
        template: WorkflowTemplate,
        requirements: WorkflowRequirements,
    ) -> Dict[str, Any]:
        """Instantiate template with requirements.

        Args:
            template: Template to instantiate
            requirements: Requirements for parameter values

        Returns:
            Complete workflow schema ready for StateGraph.from_schema()

        Raises:
            ValueError: If instantiation fails
        """
        logger.debug(f"Instantiating template: {template.name}")

        # Start with template schema
        schema = copy.deepcopy(template.schema)

        # Replace placeholders
        schema = self._replace_placeholders(schema, requirements)

        # Validate schema
        self._validate_instantiated_schema(schema)

        return schema

    def list_templates(
        self,
        vertical: Optional[str] = None,
        template_type: Optional[TemplateType] = None,
    ) -> List[WorkflowTemplate]:
        """List available templates.

        Args:
            vertical: Filter by vertical (None = all)
            template_type: Filter by type (None = all)

        Returns:
            List of matching templates
        """
        templates = self._templates

        if vertical:
            templates = [t for t in templates if vertical in t.verticals]

        if template_type:
            templates = [t for t in templates if t.template_type == template_type]

        return templates

    # =============================================================================
    # Template Definitions
    # =============================================================================

    def _load_builtin_templates(self) -> List[WorkflowTemplate]:
        """Load built-in workflow templates.

        Returns templates for all verticals and common patterns.
        """
        templates = []

        # === Research Templates ===

        templates.append(
            WorkflowTemplate(
                name="sequential_research",
                description="Sequential research workflow with analysis",
                template_type=TemplateType.SEQUENTIAL,
                verticals=["research"],
                keywords=["research", "analyze", "investigate", "study"],
                execution_order="sequential",
                min_tasks=2,
                max_tasks=5,
                schema=self._sequential_research_schema(),
            )
        )

        templates.append(
            WorkflowTemplate(
                name="conditional_research",
                description="Research with quality-based branching",
                template_type=TemplateType.CONDITIONAL,
                verticals=["research"],
                keywords=["research", "quality", "validate", "check"],
                execution_order="conditional",
                min_tasks=2,
                schema=self._conditional_research_schema(),
            )
        )

        # === Coding Templates ===

        templates.append(
            WorkflowTemplate(
                name="sequential_coding",
                description="Sequential coding workflow (analyze → implement → test)",
                template_type=TemplateType.SEQUENTIAL,
                verticals=["coding"],
                keywords=["code", "implement", "develop", "feature"],
                execution_order="sequential",
                min_tasks=2,
                max_tasks=6,
                schema=self._sequential_coding_schema(),
            )
        )

        templates.append(
            WorkflowTemplate(
                name="bug_fix_workflow",
                description="Bug fix workflow with testing loop",
                template_type=TemplateType.LOOP,
                verticals=["coding"],
                keywords=["bug", "fix", "error", "debug"],
                execution_order="conditional",
                min_tasks=2,
                schema=self._bug_fix_schema(),
            )
        )

        # === DevOps Templates ===

        templates.append(
            WorkflowTemplate(
                name="deploy_workflow",
                description="Deployment workflow with validation",
                template_type=TemplateType.SEQUENTIAL,
                verticals=["devops"],
                keywords=["deploy", "release", "ship"],
                execution_order="sequential",
                min_tasks=2,
                max_tasks=5,
                schema=self._deploy_schema(),
            )
        )

        # === Data Analysis Templates ===

        templates.append(
            WorkflowTemplate(
                name="eda_workflow",
                description="Exploratory data analysis workflow",
                template_type=TemplateType.SEQUENTIAL,
                verticals=["dataanalysis"],
                keywords=["analyze", "explore", "eda", "statistics"],
                execution_order="sequential",
                min_tasks=2,
                max_tasks=5,
                schema=self._eda_schema(),
            )
        )

        return templates

    # =============================================================================
    # Template Schemas
    # =============================================================================

    def _sequential_research_schema(self) -> Dict[str, Any]:
        """Schema for sequential research workflow."""
        return {
            "workflow_name": "{workflow_name}",
            "description": "{description}",
            "nodes": [
                {
                    "id": "gather",
                    "type": "agent",
                    "role": "researcher",
                    "goal": "Gather information on {topic}",
                    "tool_budget": 10,
                    "allowed_tools": ["search", "read"],
                    "output_key": "sources",
                },
                {
                    "id": "analyze",
                    "type": "agent",
                    "role": "analyst",
                    "goal": "Analyze gathered sources",
                    "tool_budget": 8,
                    "output_key": "analysis",
                },
                {
                    "id": "synthesize",
                    "type": "agent",
                    "role": "writer",
                    "goal": "Synthesize findings into summary",
                    "tool_budget": 5,
                    "output_key": "summary",
                },
            ],
            "edges": [
                {"source": "gather", "target": "analyze", "type": "normal"},
                {"source": "analyze", "target": "synthesize", "type": "normal"},
                {"source": "synthesize", "target": "__end__", "type": "normal"},
            ],
            "entry_point": "gather",
            "metadata": {
                "vertical": "research",
                "max_iterations": 10,
                "timeout_seconds": 300,
            },
        }

    def _conditional_research_schema(self) -> Dict[str, Any]:
        """Schema for conditional research with quality check."""
        return {
            "workflow_name": "{workflow_name}",
            "description": "{description}",
            "nodes": [
                {
                    "id": "search",
                    "type": "agent",
                    "role": "researcher",
                    "goal": "Search for information on {topic}",
                    "tool_budget": 10,
                    "allowed_tools": ["search", "read"],
                    "output_key": "results",
                },
                {
                    "id": "synthesize",
                    "type": "agent",
                    "role": "analyst",
                    "goal": "Synthesize findings",
                    "tool_budget": 5,
                    "output_key": "summary",
                },
                {
                    "id": "check_quality",
                    "type": "condition",
                    "condition": "has_sufficient_quality",
                    "branches": {
                        "sufficient": "__end__",
                        "insufficient": "search",
                    },
                },
            ],
            "edges": [
                {"source": "search", "target": "synthesize", "type": "normal"},
                {"source": "synthesize", "target": "check_quality", "type": "normal"},
                {
                    "source": "check_quality",
                    "target": {
                        "sufficient": "__end__",
                        "insufficient": "search",
                    },
                    "type": "conditional",
                    "condition": "has_sufficient_quality",
                },
            ],
            "entry_point": "search",
            "metadata": {
                "vertical": "research",
                "max_iterations": 15,
                "timeout_seconds": 400,
            },
        }

    def _sequential_coding_schema(self) -> Dict[str, Any]:
        """Schema for sequential coding workflow."""
        return {
            "workflow_name": "{workflow_name}",
            "description": "{description}",
            "nodes": [
                {
                    "id": "analyze",
                    "type": "agent",
                    "role": "researcher",
                    "goal": "Analyze the codebase",
                    "tool_budget": 8,
                    "allowed_tools": ["read", "search"],
                    "output_key": "analysis",
                },
                {
                    "id": "implement",
                    "type": "agent",
                    "role": "executor",
                    "goal": "Implement the required changes",
                    "tool_budget": 15,
                    "allowed_tools": ["write", "edit"],
                    "output_key": "changes",
                },
                {
                    "id": "test",
                    "type": "compute",
                    "handler": "run_tests",
                    "output_key": "test_results",
                },
            ],
            "edges": [
                {"source": "analyze", "target": "implement", "type": "normal"},
                {"source": "implement", "target": "test", "type": "normal"},
                {"source": "test", "target": "__end__", "type": "normal"},
            ],
            "entry_point": "analyze",
            "metadata": {
                "vertical": "coding",
                "max_iterations": 20,
                "timeout_seconds": 600,
            },
        }

    def _bug_fix_schema(self) -> Dict[str, Any]:
        """Schema for bug fix workflow with testing loop."""
        return {
            "workflow_name": "{workflow_name}",
            "description": "{description}",
            "nodes": [
                {
                    "id": "investigate",
                    "type": "agent",
                    "role": "researcher",
                    "goal": "Investigate the bug",
                    "tool_budget": 10,
                    "allowed_tools": ["read", "search", "git_log"],
                    "output_key": "findings",
                },
                {
                    "id": "fix",
                    "type": "agent",
                    "role": "executor",
                    "goal": "Implement the fix",
                    "tool_budget": 12,
                    "allowed_tools": ["write", "edit"],
                    "output_key": "fix",
                },
                {
                    "id": "test",
                    "type": "compute",
                    "handler": "run_tests",
                    "output_key": "results",
                },
                {
                    "id": "check_tests",
                    "type": "condition",
                    "condition": "tests_passed",
                    "branches": {
                        "passed": "__end__",
                        "failed": "fix",
                    },
                },
            ],
            "edges": [
                {"source": "investigate", "target": "fix", "type": "normal"},
                {"source": "fix", "target": "test", "type": "normal"},
                {"source": "test", "target": "check_tests", "type": "normal"},
                {
                    "source": "check_tests",
                    "target": {
                        "passed": "__end__",
                        "failed": "fix",
                    },
                    "type": "conditional",
                    "condition": "tests_passed",
                },
            ],
            "entry_point": "investigate",
            "metadata": {
                "vertical": "coding",
                "max_iterations": 15,
                "timeout_seconds": 500,
            },
        }

    def _deploy_schema(self) -> Dict[str, Any]:
        """Schema for deployment workflow."""
        return {
            "workflow_name": "{workflow_name}",
            "description": "{description}",
            "nodes": [
                {
                    "id": "build",
                    "type": "compute",
                    "handler": "build_artifact",
                    "output_key": "artifact",
                },
                {
                    "id": "test",
                    "type": "compute",
                    "handler": "run_integration_tests",
                    "output_key": "test_results",
                },
                {
                    "id": "deploy",
                    "type": "compute",
                    "handler": "deploy_to_environment",
                    "output_key": "deployment",
                },
                {
                    "id": "verify",
                    "type": "compute",
                    "handler": "verify_deployment",
                    "output_key": "verification",
                },
            ],
            "edges": [
                {"source": "build", "target": "test", "type": "normal"},
                {"source": "test", "target": "deploy", "type": "normal"},
                {"source": "deploy", "target": "verify", "type": "normal"},
                {"source": "verify", "target": "__end__", "type": "normal"},
            ],
            "entry_point": "build",
            "metadata": {
                "vertical": "devops",
                "max_iterations": 10,
                "timeout_seconds": 600,
            },
        }

    def _eda_schema(self) -> Dict[str, Any]:
        """Schema for exploratory data analysis."""
        return {
            "workflow_name": "{workflow_name}",
            "description": "{description}",
            "nodes": [
                {
                    "id": "load_data",
                    "type": "compute",
                    "handler": "load_dataset",
                    "output_key": "data",
                },
                {
                    "id": "analyze",
                    "type": "compute",
                    "handler": "compute_statistics",
                    "output_key": "stats",
                },
                {
                    "id": "visualize",
                    "type": "compute",
                    "handler": "create_visualizations",
                    "output_key": "plots",
                },
            ],
            "edges": [
                {"source": "load_data", "target": "analyze", "type": "normal"},
                {"source": "analyze", "target": "visualize", "type": "normal"},
                {"source": "visualize", "target": "__end__", "type": "normal"},
            ],
            "entry_point": "load_data",
            "metadata": {
                "vertical": "dataanalysis",
                "max_iterations": 10,
                "timeout_seconds": 300,
            },
        }

    # =============================================================================
    # Helper Methods
    # =============================================================================

    def _replace_placeholders(
        self, schema: Dict[str, Any], requirements: WorkflowRequirements
    ) -> Dict[str, Any]:
        """Replace placeholders in template with values from requirements.

        Supports placeholders like:
        - {workflow_name} -> requirements.description[:50]
        - {description} -> requirements.description
        - {topic} -> extracted from description

        Args:
            schema: Template schema with placeholders
            requirements: Requirements for values

        Returns:
            Schema with placeholders replaced
        """
        import copy

        schema = copy.deepcopy(schema)

        # Extract basic values
        workflow_name = (
            requirements.description.lower().replace(" ", "_")[:50]
            if requirements.description
            else "generated_workflow"
        )

        # Extract topic from first task or description
        topic = (
            requirements.functional.tasks[0].description
            if requirements.functional.tasks
            else requirements.description
        )

        replacements = {
            "{workflow_name}": workflow_name,
            "{description}": requirements.description,
            "{topic}": topic,
        }

        # Replace in strings
        def replace_in_dict(obj: Any) -> Any:
            if isinstance(obj, str):
                for placeholder, value in replacements.items():
                    obj = obj.replace(placeholder, str(value))
                return obj
            elif isinstance(obj, dict):
                return {k: replace_in_dict(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_in_dict(item) for item in obj]
            return obj

        return replace_in_dict(schema)

    def _validate_instantiated_schema(self, schema: Dict[str, Any]) -> None:
        """Validate instantiated schema.

        Args:
            schema: Instantiated schema

        Raises:
            ValueError: If schema is invalid
        """
        required_fields = ["nodes", "edges", "entry_point"]
        for field_name in required_fields:
            if field_name not in schema:
                raise ValueError(f"Invalid schema: missing '{field_name}'")

        # Check for remaining placeholders (only our specific placeholders)
        schema_str = json.dumps(schema)
        expected_placeholders = ["{workflow_name}", "{description}", "{topic}"]
        found_placeholders = [p for p in expected_placeholders if p in schema_str]
        if found_placeholders:
            raise ValueError(f"Schema still contains placeholders: {set(found_placeholders)}")


# =============================================================================
# Utilities
# =============================================================================


import copy

__all__ = [
    "TemplateLibrary",
    "WorkflowTemplate",
    "TemplateType",
]
