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

"""Generic persona templates for cross-vertical reuse.

This module provides base persona templates that can be customized
by verticals for domain-specific behavior.

Design Pattern: Template Method
- Base personas with common traits
- Verticals extend with domain-specific expertise
- Consistent persona interface across verticals

Example:
    from victor.framework.multi_agent.persona_templates import get_researcher_template

    # Get base researcher template
    base_researcher = get_researcher_template()

    # Customize for coding vertical
    coding_researcher = base_researcher.model_copy(
        expertise=["code_architecture", "design_patterns"],
        communication_style="concise_technical"
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from victor.framework.multi_agent.personas import (
    CommunicationStyle,
    ExpertiseLevel,
    PersonaTraits,
)

# =============================================================================
# Persona Templates
# =============================================================================


def get_researcher_template() -> PersonaTraits:
    """Get base researcher persona template.

    Returns:
        PersonaTraits for researcher role
    """
    return PersonaTraits(
        name="researcher",
        role="Research Specialist",
        description="Expert at gathering, analyzing, and synthesizing information from diverse sources",
        communication_style=CommunicationStyle.TECHNICAL,
        expertise_level=ExpertiseLevel.EXPERT,
        strengths=[
            "information_gathering",
            "source_verification",
            "synthesis",
            "analysis",
        ],
        verbosity=0.7,
        custom_traits={
            "traits": ["thorough", "systematic", "evidence_based", "objective"],
            "prompt_extensions": {
                "focus": "Comprehensive research from multiple sources",
                "approach": "Verify claims and cite sources",
                "output": "Detailed analysis with references",
            },
        },
    )


def get_planner_template() -> PersonaTraits:
    """Get base planner persona template.

    Returns:
        PersonaTraits for planner role
    """
    return PersonaTraits(
        name="planner",
        role="Planning Specialist",
        description="Expert at breaking down complex tasks and creating structured plans",
        communication_style=CommunicationStyle.TECHNICAL,
        expertise_level=ExpertiseLevel.EXPERT,
        strengths=[
            "task_breakdown",
            "architecture",
            "sequencing",
            "dependency_analysis",
        ],
        verbosity=0.6,
        custom_traits={
            "traits": ["strategic", "methodical", "forward_thinking", "risk_aware"],
            "prompt_extensions": {
                "focus": "Structured approach with clear steps",
                "approach": "Consider dependencies and risks",
                "output": "Detailed plans with milestones",
            },
        },
    )


def get_executor_template() -> PersonaTraits:
    """Get base executor persona template.

    Returns:
        PersonaTraits for executor role
    """
    return PersonaTraits(
        name="executor",
        role="Execution Specialist",
        description="Expert at implementing solutions efficiently and correctly",
        communication_style=CommunicationStyle.CONCISE,
        expertise_level=ExpertiseLevel.EXPERT,
        strengths=["implementation", "debugging", "testing", "optimization"],
        verbosity=0.4,
        custom_traits={
            "traits": ["focused", "efficient", "quality_conscious", "pragmatic"],
            "prompt_extensions": {
                "focus": "Correct and efficient implementation",
                "approach": "Test thoroughly and handle edge cases",
                "output": "Working code with validation",
            },
        },
    )


def get_reviewer_template() -> PersonaTraits:
    """Get base reviewer persona template.

    Returns:
        PersonaTraits for reviewer role
    """
    return PersonaTraits(
        name="reviewer",
        role="Review Specialist",
        description="Expert at evaluating quality, identifying issues, and suggesting improvements",
        communication_style=CommunicationStyle.FORMAL,
        expertise_level=ExpertiseLevel.SPECIALIST,
        strengths=["code_review", "quality_assessment", "best_practices", "security"],
        verbosity=0.6,
        custom_traits={
            "traits": [
                "detail_oriented",
                "critical_thinking",
                "standards_driven",
                "helpful",
            ],
            "prompt_extensions": {
                "focus": "Quality and best practices",
                "approach": "Identify issues and suggest improvements",
                "output": "Constructive feedback with recommendations",
            },
        },
    )


# =============================================================================
# Template Registry
# =============================================================================


PERSONA_TEMPLATES: Dict[str, PersonaTraits] = {
    "researcher": get_researcher_template(),
    "planner": get_planner_template(),
    "executor": get_executor_template(),
    "reviewer": get_reviewer_template(),
}


def get_persona_template(name: str) -> Optional[PersonaTraits]:
    """Get a persona template by name.

    Args:
        name: Template name (researcher, planner, executor, reviewer)

    Returns:
        PersonaTraits template or None if not found
    """
    return PERSONA_TEMPLATES.get(name)


def list_persona_templates() -> List[str]:
    """List all available persona templates.

    Returns:
        List of template names
    """
    return list(PERSONA_TEMPLATES.keys())


__all__ = [
    "get_researcher_template",
    "get_planner_template",
    "get_executor_template",
    "get_reviewer_template",
    "get_persona_template",
    "list_persona_templates",
    "PERSONA_TEMPLATES",
]
