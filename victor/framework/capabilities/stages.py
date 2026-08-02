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

"""Stage builder capability for framework-level workflow stages.

This module provides reusable stage definitions with a 7-stage generic workflow
template (INITIAL → COMPLETION) that verticals can use and customize.

Design Pattern: Capability Provider
- Generic stage templates for common workflows
- Stage validation and transition rules
- Stage-to-tool mappings
- Prompt hints per stage

Integration Point:
    Update VerticalBase.get_stages() to use StageBuilderCapability by default

Example:
    capability = StageBuilderCapability()
    stages = capability.get_stages()

    # Customize for specific vertical
    custom_stages = capability.get_custom_stages({
        "PLANNING": {"tools": {"read", "grep"}},
        "EXECUTION": {"tools": {"write", "edit", "shell"}},
    })

Phase 1: Promote Generic Capabilities to Framework
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from enum import Enum


class StandardStage(Enum):
    """Standard stage names for the generic workflow template.

    The 7-stage workflow represents a complete problem-solving lifecycle:
    1. INITIAL: Understand the request and gather initial context
    2. PLANNING: Design the approach and strategy
    3. READING: Gather detailed information and context
    4. ANALYSIS: Analyze information and identify solutions
    5. EXECUTION: Implement the planned changes or actions
    6. VERIFICATION: Validate results and test outcomes
    7. COMPLETION: Finalize, document, and wrap up
    """

    INITIAL = "INITIAL"
    PLANNING = "PLANNING"
    READING = "READING"
    ANALYSIS = "ANALYSIS"
    EXECUTION = "EXECUTION"
    VERIFICATION = "VERIFICATION"
    COMPLETION = "COMPLETION"


@dataclass
class StagePromptHint:
    """Prompt hint for a specific stage.

    Attributes:
        stage: Stage name this hint applies to
        hint_text: Text to include in system prompt for this stage
        tool_priorities: Tools to prioritize in this stage
        constraints: Any constraints for this stage
    """

    stage: str
    hint_text: str
    tool_priorities: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stage": self.stage,
            "hint_text": self.hint_text,
            "tool_priorities": self.tool_priorities,
            "constraints": self.constraints,
        }


class StageBuilderCapability:
    """Generic stage builder capability provider.

    Provides reusable stage definitions for vertical workflows. Verticals can
    use the default 7-stage workflow template or customize specific stages
    for domain-specific needs.

    The workflow applies to most domains:
    - Coding: understand -> plan -> read code -> analyze -> implement -> test -> commit
    - DevOps: assess -> plan -> implement -> validate -> deploy -> monitor
    - Research: question -> search -> read -> synthesize -> write -> verify
    - Data Analysis: understand -> explore -> analyze -> visualize -> report

    Attributes:
        custom_stages: Optional custom stage definitions
        enable_validation: Whether to validate stage transitions
    """

    # Default stage definitions (7-stage workflow template)
    DEFAULT_STAGES: Dict[str, Dict[str, Any]] = {
        "INITIAL": {
            "name": "INITIAL",
            "description": "Understanding the request and gathering initial context",
            "keywords": [
                "what",
                "how",
                "explain",
                "help",
                "where",
                "show me",
                "describe",
                "overview",
                "understand",
                "clarify",
            ],
            "next_stages": {"PLANNING", "READING"},
            "prompt_hint": "Start by understanding what the user is asking for. "
            "Ask clarifying questions if needed.",
        },
        "PLANNING": {
            "name": "PLANNING",
            "description": "Designing the approach and creating a strategy",
            "keywords": [
                "plan",
                "approach",
                "strategy",
                "design",
                "architecture",
                "outline",
                "steps",
                "roadmap",
                "how should",
                "what's the best way",
            ],
            "next_stages": {"READING", "EXECUTION"},
            "prompt_hint": "Create a clear plan with specific steps. "
            "Break down complex tasks into manageable parts.",
        },
        "READING": {
            "name": "READING",
            "description": "Gathering detailed information and context",
            "keywords": [
                "read",
                "show",
                "find",
                "search",
                "look",
                "check",
                "examine",
                "inspect",
                "review",
                "fetch",
                "get",
                "retrieve",
            ],
            "next_stages": {"ANALYSIS", "EXECUTION"},
            "prompt_hint": "Gather all necessary information before proceeding. "
            "Read relevant files and documentation.",
        },
        "ANALYSIS": {
            "name": "ANALYSIS",
            "description": "Analyzing information and identifying solutions",
            "keywords": [
                "analyze",
                "review",
                "understand",
                "why",
                "how does",
                "compare",
                "evaluate",
                "assess",
                "investigate",
                "diagnose",
            ],
            "next_stages": {"EXECUTION", "PLANNING"},
            "prompt_hint": "Analyze the information carefully. "
            "Identify patterns, root causes, and potential solutions.",
        },
        "EXECUTION": {
            "name": "EXECUTION",
            "description": "Implementing the planned changes or actions",
            "keywords": [
                "change",
                "modify",
                "create",
                "add",
                "remove",
                "fix",
                "implement",
                "write",
                "update",
                "refactor",
                "build",
                "configure",
                "set up",
                "install",
                "run",
                "execute",
            ],
            "next_stages": {"VERIFICATION", "COMPLETION"},
            "prompt_hint": "Implement the solution carefully. "
            "Follow best practices and existing patterns.",
        },
        "VERIFICATION": {
            "name": "VERIFICATION",
            "description": "Validating results and testing outcomes",
            "keywords": [
                "test",
                "verify",
                "check",
                "validate",
                "confirm",
                "ensure",
                "run tests",
                "build",
                "compile",
                "lint",
            ],
            "next_stages": {"COMPLETION", "EXECUTION"},
            "prompt_hint": "Verify that your solution works correctly. "
            "Run tests and check for errors.",
        },
        "COMPLETION": {
            "name": "COMPLETION",
            "description": "Finalizing, documenting, and wrapping up",
            "keywords": [
                "done",
                "finish",
                "complete",
                "commit",
                "summarize",
                "document",
                "conclude",
                "wrap up",
                "finalize",
            ],
            "next_stages": set(),
            "prompt_hint": "Summarize what was accomplished. "
            "Document any important changes or decisions.",
        },
    }

    # Stage-to-tool mappings for common task types
    STAGE_TOOL_MAPPINGS: Dict[str, Set[str]] = {
        "INITIAL": {"read", "ls", "grep"},
        "PLANNING": {"read", "grep", "ls"},
        "READING": {"read", "grep", "search", "web_search", "web_fetch"},
        "ANALYSIS": {"read", "grep", "search"},
        "EXECUTION": {"write", "edit", "shell", "git"},
        "VERIFICATION": {"shell", "test", "lint"},
        "COMPLETION": {"git", "write"},
    }

    def __init__(
        self,
        custom_stages: Optional[Dict[str, Dict[str, Any]]] = None,
        enable_validation: bool = True,
    ):
        """Initialize the stage builder capability.

        Args:
            custom_stages: Optional custom stage definitions to override defaults
            enable_validation: Whether to validate stage transitions
        """
        self._custom_stages = custom_stages or {}
        self._enable_validation = enable_validation
        self._stage_cache: Optional[Dict[str, Any]] = None

    def get_stages(self) -> Dict[str, Any]:
        """Get all stage definitions.

        Returns:
            Dictionary mapping stage names to stage definitions
        """
        if self._stage_cache is not None:
            return self._stage_cache.copy()

        # Merge defaults with custom stages
        stages = self.DEFAULT_STAGES.copy()
        stages.update(self._custom_stages)

        # Validate if enabled
        if self._enable_validation:
            self._validate_stages(stages)

        self._stage_cache = stages
        return stages.copy()

    def get_custom_stages(self, overrides: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Get customized stage definitions with overrides.

        Args:
            overrides: Dictionary of stage names to override properties

        Returns:
            Dictionary of customized stage definitions
        """
        stages = self.DEFAULT_STAGES.copy()

        for stage_name, override in overrides.items():
            if stage_name in stages:
                # Update existing stage
                stages[stage_name].update(override)
            else:
                # Add new custom stage
                stages[stage_name] = override

        # Validate if enabled
        if self._enable_validation:
            self._validate_stages(stages)

        return stages

    def get_stage(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific stage definition.

        Args:
            stage_name: Name of the stage to retrieve

        Returns:
            Stage definition or None if not found
        """
        stages = self.get_stages()
        return stages.get(stage_name)

    def get_prompt_hints(self) -> List[StagePromptHint]:
        """Get prompt hints for all stages.

        Returns:
            List of StagePromptHint objects
        """
        stages = self.get_stages()
        hints = []

        for stage_name, stage_def in stages.items():
            hint = StagePromptHint(
                stage=stage_name,
                hint_text=stage_def.get("prompt_hint", ""),
                tool_priorities=list(self.STAGE_TOOL_MAPPINGS.get(stage_name, set())),
            )
            hints.append(hint)

        return hints

    def get_tools_for_stage(self, stage_name: str) -> Set[str]:
        """Get recommended tools for a specific stage.

        Args:
            stage_name: Name of the stage

        Returns:
            Set of tool names recommended for this stage
        """
        return self.STAGE_TOOL_MAPPINGS.get(stage_name, set())

    def get_valid_transitions(self, stage_name: str) -> Set[str]:
        """Get valid next stages for a given stage.

        Args:
            stage_name: Current stage name

        Returns:
            Set of valid next stage names
        """
        stage = self.get_stage(stage_name)
        if stage:
            return set(stage.get("next_stages", set()))
        return set()

    def is_valid_transition(self, from_stage: str, to_stage: str) -> bool:
        """Check if a stage transition is valid.

        Args:
            from_stage: Current stage name
            to_stage: Next stage name

        Returns:
            True if transition is valid, False otherwise
        """
        valid_stages = self.get_valid_transitions(from_stage)
        return to_stage in valid_stages

    def _validate_stages(self, stages: Dict[str, Dict[str, Any]]) -> None:
        """Validate stage definitions.

        Args:
            stages: Dictionary of stage definitions to validate

        Raises:
            ValueError: If validation fails
        """
        # Check for required stages
        required_stages = {"INITIAL", "COMPLETION"}
        missing_stages = required_stages - set(stages.keys())
        if missing_stages:
            raise ValueError(
                f"Missing required stages: {missing_stages}. "
                "All workflows must have INITIAL and COMPLETION stages."
            )

        # Validate stage transitions
        for stage_name, stage_def in stages.items():
            next_stages = stage_def.get("next_stages", set())
            for next_stage in next_stages:
                if next_stage not in stages:
                    raise ValueError(
                        f"Invalid transition from {stage_name} to {next_stage}: "
                        f"stage {next_stage} does not exist"
                    )

    def get_workflow_summary(self) -> str:
        """Get a human-readable summary of the workflow.

        Returns:
            Summary string describing the workflow stages
        """
        stages = self.get_stages()
        summary_parts = []

        for stage_name in [
            "INITIAL",
            "PLANNING",
            "READING",
            "ANALYSIS",
            "EXECUTION",
            "VERIFICATION",
            "COMPLETION",
        ]:
            if stage_name in stages:
                stage = stages[stage_name]
                summary_parts.append(f"{stage_name}: {stage.get('description', '')}")

        return "\n".join(summary_parts)

    def clear_cache(self) -> None:
        """Clear the stage cache."""
        self._stage_cache = None


# Pre-configured stage builders for common vertical types
class StageBuilderPresets:
    """Pre-configured stage builders for common verticals."""

    @staticmethod
    def coding() -> StageBuilderCapability:
        """Get stage builder optimized for coding vertical."""
        custom_stages = {
            "ANALYSIS": {
                "description": "Analyzing code structure and identifying solutions",
                "keywords": ["analyze", "understand", "debug", "trace"],
            },
            "EXECUTION": {
                "description": "Writing and modifying code",
                "keywords": ["write", "edit", "refactor", "implement"],
            },
        }
        return StageBuilderCapability(custom_stages=custom_stages)

    @staticmethod
    def devops() -> StageBuilderCapability:
        """Get stage builder optimized for DevOps vertical."""
        custom_stages = {
            "READING": {
                "description": "Reviewing infrastructure and configuration",
                "keywords": ["review", "check", "inspect"],
            },
            "EXECUTION": {
                "description": "Deploying and configuring infrastructure",
                "keywords": ["deploy", "configure", "provision"],
            },
        }
        return StageBuilderCapability(custom_stages=custom_stages)

    @staticmethod
    def research() -> StageBuilderCapability:
        """Get stage builder optimized for research vertical."""
        custom_stages = {
            "READING": {
                "description": "Searching and reading sources",
                "keywords": ["search", "read", "find", "investigate"],
            },
            "ANALYSIS": {
                "description": "Synthesizing findings from sources",
                "keywords": ["synthesize", "compare", "evaluate"],
            },
        }
        return StageBuilderCapability(custom_stages=custom_stages)

    @staticmethod
    def data_analysis() -> StageBuilderCapability:
        """Get stage builder optimized for data analysis vertical."""
        custom_stages = {
            "READING": {
                "description": "Loading and exploring data",
                "keywords": ["load", "explore", "inspect"],
            },
            "ANALYSIS": {
                "description": "Analyzing data and computing metrics",
                "keywords": ["analyze", "compute", "calculate"],
            },
        }
        return StageBuilderCapability(custom_stages=custom_stages)


__all__ = [
    "StageBuilderCapability",
    "StageBuilderPresets",
    "StandardStage",
    "StagePromptHint",
]
