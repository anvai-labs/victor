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

"""Persona System for agent characterization.

This module provides a persona system that allows agents to have consistent
personality traits, communication styles, and expertise areas. Personas help
create more natural and engaging interactions.

Example:
    from victor.framework.personas import Persona, get_persona, register_persona

    # Use a built-in persona
    friendly = get_persona("friendly_assistant")
    prompt = friendly.get_system_prompt_section()

    # Create a custom persona
    custom = Persona(
        name="Security Expert",
        background="You are a cybersecurity specialist with 15 years experience.",
        communication_style="formal",
        expertise_areas=("security", "authentication", "encryption"),
        quirks=("always considers edge cases", "emphasizes best practices"),
    )
    register_persona("security_expert", custom)

    # Format messages with persona style
    message = custom.format_message("hello there")  # "Hello there."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Persona:
    """A personality profile for an agent.

    Personas define consistent character traits, communication styles,
    and expertise areas that shape how an agent interacts. They can be
    used to create more engaging and natural conversations.

    Attributes:
        name: Display name for this persona
        background: Rich description of persona's history and character
        communication_style: How the persona communicates (e.g., formal, casual)
        expertise_areas: Domains of expertise (e.g., python, security)
        quirks: Personality quirks and behaviors

    Example:
        persona = Persona(
            name="Senior Developer",
            background="You are a senior software engineer with expertise in Python.",
            communication_style="professional",
            expertise_areas=("python", "architecture", "best-practices"),
            quirks=("asks clarifying questions", "suggests alternatives"),
        )
    """

    name: str
    background: str
    communication_style: str = "professional"
    expertise_areas: Tuple[str, ...] = ()
    quirks: Tuple[str, ...] = ()

    def format_message(self, content: str) -> str:
        """Format a message according to persona's communication style.

        Applies style-specific formatting to the message content.

        Args:
            content: The raw message content

        Returns:
            Formatted message according to communication style

        Example:
            formal_persona.format_message("hello")  # "Hello."
            casual_persona.format_message("HELLO")  # "hello"
        """
        if not content:
            return content

        style = self.communication_style.lower()

        if style == "formal":
            # Capitalize first letter and ensure punctuation
            result = content[0].upper() + content[1:] if len(content) > 1 else content.upper()
            if not result.endswith((".", "!", "?")):
                result = result + "."
            return result

        elif style == "casual":
            # Lowercase everything for casual style
            return content.lower()

        else:
            # Professional and other styles preserve original
            return content

    def get_system_prompt_section(self) -> str:
        """Generate a system prompt section for this persona.

        Creates a formatted text block that can be included in a
        system prompt to establish the persona's character.

        Returns:
            Formatted system prompt section

        Example:
            prompt = persona.get_system_prompt_section()
            full_prompt = f"You are an AI assistant.\\n\\n{prompt}"
        """
        sections = []

        # Identity section
        sections.append(f"## Persona: {self.name}")
        sections.append("")
        sections.append(self.background)

        # Communication style
        sections.append("")
        sections.append(f"**Communication Style**: {self.communication_style}")

        # Expertise areas
        if self.expertise_areas:
            areas = ", ".join(self.expertise_areas)
            sections.append(f"**Expertise Areas**: {areas}")

        # Quirks and behaviors
        if self.quirks:
            sections.append("")
            sections.append("**Behavioral Traits**:")
            for quirk in self.quirks:
                sections.append(f"- {quirk}")

        return "\n".join(sections)

    def to_dict(self) -> Dict[str, Any]:
        """Convert persona to dictionary for serialization.

        Returns:
            Dictionary representation of the persona
        """
        return {
            "name": self.name,
            "background": self.background,
            "communication_style": self.communication_style,
            "expertise_areas": self.expertise_areas,
            "quirks": self.quirks,
        }


# =============================================================================
# Built-in Persona Registry
# =============================================================================


PERSONA_REGISTRY: Dict[str, Persona] = {
    "friendly_assistant": Persona(
        name="Friendly Assistant",
        background=(
            "You are a warm, helpful assistant who genuinely enjoys helping people. "
            "You communicate in a friendly, approachable manner while remaining "
            "professional and accurate. You celebrate successes and provide "
            "encouragement when users face challenges."
        ),
        communication_style="friendly",
        expertise_areas=("general assistance", "problem solving", "guidance"),
        quirks=(
            "uses encouraging language",
            "asks if clarification is needed",
            "celebrates small wins",
        ),
    ),
    "senior_developer": Persona(
        name="Senior Developer",
        background=(
            "You are an experienced senior software developer with deep expertise "
            "in multiple programming languages and software architecture. You've "
            "seen many codebases evolve and understand the importance of clean, "
            "maintainable code. You mentor junior developers and advocate for "
            "best practices while being pragmatic about trade-offs."
        ),
        communication_style="professional",
        expertise_areas=(
            "software architecture",
            "code quality",
            "design patterns",
            "performance optimization",
            "debugging",
        ),
        quirks=(
            "suggests refactoring opportunities",
            "considers edge cases",
            "explains the 'why' behind recommendations",
            "balances ideal solutions with practical constraints",
        ),
    ),
    "code_reviewer": Persona(
        name="Code Reviewer",
        background=(
            "You are a meticulous code reviewer who ensures code quality, "
            "consistency, and adherence to best practices. You provide "
            "constructive feedback that helps developers improve their skills. "
            "You balance thoroughness with pragmatism, focusing on issues that "
            "matter while not being pedantic about minor stylistic preferences."
        ),
        communication_style="professional",
        expertise_areas=(
            "code review",
            "code quality",
            "security",
            "testing",
            "documentation",
        ),
        quirks=(
            "provides specific, actionable feedback",
            "explains rationale for suggestions",
            "highlights both issues and good practices",
            "prioritizes feedback by severity",
        ),
    ),
    "mentor": Persona(
        name="Mentor",
        background=(
            "You are a patient, knowledgeable mentor who helps developers grow "
            "their skills. You explain concepts clearly, use analogies to make "
            "complex topics accessible, and encourage learning by doing. You "
            "remember that everyone was a beginner once and create a safe space "
            "for asking questions."
        ),
        communication_style="supportive",
        expertise_areas=(
            "teaching",
            "explaining concepts",
            "career guidance",
            "skill development",
        ),
        quirks=(
            "uses analogies and examples",
            "breaks complex topics into digestible pieces",
            "asks guiding questions rather than giving direct answers",
            "celebrates learning progress",
            "shares relevant resources and further reading",
        ),
    ),
}


# =============================================================================
# Registry Functions
# =============================================================================


def get_persona(name: str) -> Optional[Persona]:
    """Get a persona by name from the registry.

    Args:
        name: Name of the persona to retrieve

    Returns:
        Persona if found, None otherwise

    Example:
        mentor = get_persona("mentor")
        if mentor:
            prompt = mentor.get_system_prompt_section()
    """
    return PERSONA_REGISTRY.get(name)


def register_persona(name: str, persona: Persona) -> None:
    """Register a custom persona in the registry.

    Overwrites any existing persona with the same name.

    Args:
        name: Registry key for the persona
        persona: Persona instance to register

    Example:
        custom = Persona(name="Custom", background="Custom persona.")
        register_persona("custom", custom)
    """
    PERSONA_REGISTRY[name] = persona


def list_personas() -> List[str]:
    """List all registered persona names.

    Returns:
        List of registered persona names

    Example:
        names = list_personas()
        for name in names:
            persona = get_persona(name)
            print(f"{name}: {persona.name}")
    """
    return list(PERSONA_REGISTRY.keys())


__all__ = [
    "Persona",
    "PERSONA_REGISTRY",
    "get_persona",
    "register_persona",
    "list_personas",
]
