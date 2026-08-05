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

"""Enhanced persona definitions for coding team members.

This module provides rich persona configurations for coding-specific
team roles, extending the framework's PersonaTraits with:

- Structured expertise categories
- Communication style traits (extended for coding contexts)
- Decision-making preferences
- Collaboration patterns

The personas are designed to improve agent behavior through more
nuanced context injection and role-specific guidance.

Example:
    from victor_coding.teams.personas import (
        get_persona,
        CODING_PERSONAS,
        apply_persona_to_spec,
    )

    # Get a persona by role
    researcher_persona = get_persona("researcher")
    print(researcher_persona.expertise)  # ['code_analysis', 'patterns', ...]

    # Apply persona to TeamMemberSpec
    enhanced_spec = apply_persona_to_spec(spec, "researcher")

Note:
    This module uses the framework's PersonaTraits as a base and extends it
    with coding-specific traits. The CodingPersonaTraits class provides
    additional fields for coding contexts while maintaining compatibility
    with the framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

# Import framework types for base functionality
from victor_contracts import (
    CommunicationStyle as FrameworkCommunicationStyle,
    ExpertiseLevel,
    PersonaTemplate,
    PersonaTraits as FrameworkPersonaTraits,
)
from victor_contracts import PersonaRegistryProtocol, get_default_persona_registry


class ExpertiseCategory(str, Enum):
    """Categories of expertise for coding roles.

    These categories help agents understand what areas
    they should focus on during their tasks.
    """

    # Analysis expertise
    CODE_ANALYSIS = "code_analysis"
    PATTERN_RECOGNITION = "pattern_recognition"
    DEPENDENCY_ANALYSIS = "dependency_analysis"
    ARCHITECTURE = "architecture"
    SECURITY = "security"

    # Implementation expertise
    CODE_WRITING = "code_writing"
    REFACTORING = "refactoring"
    TESTING = "testing"
    DEBUGGING = "debugging"
    OPTIMIZATION = "optimization"

    # Collaboration expertise
    COMMUNICATION = "communication"
    DOCUMENTATION = "documentation"
    MENTORING = "mentoring"
    CODE_REVIEW = "code_review"

    # Domain expertise
    FRONTEND = "frontend"
    BACKEND = "backend"
    DEVOPS = "devops"
    DATABASE = "database"
    API_DESIGN = "api_design"


class CommunicationStyle(str, Enum):
    """Communication styles for coding persona characterization.

    This extends the framework's CommunicationStyle with additional
    styles specific to coding team contexts.

    Note:
        For interoperability with the framework, use to_framework_style()
        to convert to FrameworkCommunicationStyle when needed.
    """

    CONCISE = "concise"  # Brief, to-the-point
    DETAILED = "detailed"  # Thorough explanations
    SOCRATIC = "socratic"  # Question-based guidance
    ASSERTIVE = "assertive"  # Direct and confident
    COLLABORATIVE = "collaborative"  # Team-oriented
    ANALYTICAL = "analytical"  # Data-driven
    SUPPORTIVE = "supportive"  # Encouraging

    def to_framework_style(self) -> FrameworkCommunicationStyle:
        """Convert to framework CommunicationStyle.

        Maps coding-specific styles to the closest framework equivalent.

        Returns:
            Corresponding FrameworkCommunicationStyle value
        """
        mapping = {
            CommunicationStyle.CONCISE: FrameworkCommunicationStyle.CONCISE,
            CommunicationStyle.DETAILED: FrameworkCommunicationStyle.FORMAL,
            CommunicationStyle.SOCRATIC: FrameworkCommunicationStyle.TECHNICAL,
            CommunicationStyle.ASSERTIVE: FrameworkCommunicationStyle.FORMAL,
            CommunicationStyle.COLLABORATIVE: FrameworkCommunicationStyle.CASUAL,
            CommunicationStyle.ANALYTICAL: FrameworkCommunicationStyle.TECHNICAL,
            CommunicationStyle.SUPPORTIVE: FrameworkCommunicationStyle.CASUAL,
        }
        return mapping.get(self, FrameworkCommunicationStyle.TECHNICAL)


class DecisionStyle(str, Enum):
    """Decision-making styles for personas."""

    CONSERVATIVE = "conservative"  # Prefer safe, proven approaches
    PROGRESSIVE = "progressive"  # Open to new solutions
    PRAGMATIC = "pragmatic"  # Balance of both
    PERFECTIONIST = "perfectionist"  # High quality standards
    ITERATIVE = "iterative"  # Start simple, improve


@dataclass
class CodingPersonaTraits:
    """Coding-specific behavioral traits for a persona.

    This class provides coding-specific trait extensions that complement
    the framework's PersonaTraits. Use this when you need coding-specific
    attributes like decision_style and attention_to_detail.

    Attributes:
        communication_style: Primary communication approach (coding-specific enum)
        decision_style: How decisions are made
        attention_to_detail: 0.0-1.0 scale of thoroughness
        risk_tolerance: 0.0-1.0 scale of risk acceptance
        collaboration_preference: 0.0-1.0 scale (0=solo, 1=collaborative)
        verbosity: 0.0-1.0 scale of output detail
    """

    communication_style: CommunicationStyle = CommunicationStyle.COLLABORATIVE
    decision_style: DecisionStyle = DecisionStyle.PRAGMATIC
    attention_to_detail: float = 0.7
    risk_tolerance: float = 0.5
    collaboration_preference: float = 0.7
    verbosity: float = 0.5

    def to_prompt_hints(self) -> str:
        """Convert traits to prompt hints.

        Returns:
            String of behavioral hints for prompt injection
        """
        hints = []

        # Communication style hints
        style_hints = {
            CommunicationStyle.CONCISE: "Keep responses brief and focused.",
            CommunicationStyle.DETAILED: "Provide thorough explanations.",
            CommunicationStyle.SOCRATIC: "Ask clarifying questions when appropriate.",
            CommunicationStyle.ASSERTIVE: "Be direct and confident in recommendations.",
            CommunicationStyle.COLLABORATIVE: "Seek input and build on others' ideas.",
            CommunicationStyle.ANALYTICAL: "Support conclusions with data and evidence.",
            CommunicationStyle.SUPPORTIVE: "Encourage and acknowledge good approaches.",
        }
        hints.append(style_hints.get(self.communication_style, ""))

        # Decision style hints
        if self.decision_style == DecisionStyle.CONSERVATIVE:
            hints.append("Prefer proven, well-tested approaches.")
        elif self.decision_style == DecisionStyle.PROGRESSIVE:
            hints.append("Consider modern solutions and best practices.")
        elif self.decision_style == DecisionStyle.PERFECTIONIST:
            hints.append("Maintain high quality standards.")
        elif self.decision_style == DecisionStyle.ITERATIVE:
            hints.append("Start simple and iterate to improve.")

        # Detail level
        if self.attention_to_detail > 0.8:
            hints.append("Pay close attention to edge cases and details.")
        elif self.attention_to_detail < 0.3:
            hints.append("Focus on the big picture over minor details.")

        # Risk tolerance
        if self.risk_tolerance < 0.3:
            hints.append("Avoid risky changes without thorough testing.")
        elif self.risk_tolerance > 0.7:
            hints.append("Don't be afraid to try unconventional solutions.")

        return " ".join(h for h in hints if h)

    def to_framework_traits(
        self,
        name: str,
        role: str,
        description: str,
        strengths: Optional[List[str]] = None,
        preferred_tools: Optional[List[str]] = None,
    ) -> FrameworkPersonaTraits:
        """Convert to framework PersonaTraits.

        Creates a framework-compatible PersonaTraits instance from
        the coding-specific traits.

        Args:
            name: Display name for the persona
            role: Role identifier
            description: Description of the persona
            strengths: Optional list of strengths
            preferred_tools: Optional list of preferred tools

        Returns:
            FrameworkPersonaTraits instance
        """
        return FrameworkPersonaTraits(
            name=name,
            role=role,
            description=description,
            communication_style=self.communication_style.to_framework_style(),
            expertise_level=ExpertiseLevel.EXPERT,
            verbosity=self.verbosity,
            strengths=strengths or [],
            preferred_tools=preferred_tools or [],
            risk_tolerance=self.risk_tolerance,
            creativity=1.0 - self.attention_to_detail,  # Map attention to creativity
            custom_traits={
                "decision_style": self.decision_style.value,
                "attention_to_detail": self.attention_to_detail,
                "collaboration_preference": self.collaboration_preference,
            },
        )


# Backward compatibility alias
PersonaTraits = CodingPersonaTraits


@dataclass
class CodingPersona:
    """Complete persona definition for a coding role.

    This combines expertise areas, personality traits, and
    role-specific guidance into a comprehensive persona.

    Attributes:
        name: Display name for the persona
        role: Base role (researcher, planner, executor, reviewer)
        expertise: Primary areas of expertise
        secondary_expertise: Secondary/supporting expertise
        traits: Behavioral traits
        strengths: Key strengths in bullet points
        approach: How this persona approaches work
        communication_patterns: Typical communication patterns
        working_style: Description of working approach
    """

    name: str
    role: str
    expertise: List[ExpertiseCategory]
    secondary_expertise: List[ExpertiseCategory] = field(default_factory=list)
    traits: PersonaTraits = field(default_factory=PersonaTraits)
    strengths: List[str] = field(default_factory=list)
    approach: str = ""
    communication_patterns: List[str] = field(default_factory=list)
    working_style: str = ""

    def get_expertise_list(self) -> List[str]:
        """Get combined expertise as string list.

        Returns:
            List of expertise category values
        """
        all_expertise = self.expertise + self.secondary_expertise
        return [e.value for e in all_expertise]

    def generate_backstory(self) -> str:
        """Generate a rich backstory from persona attributes.

        Returns:
            Multi-sentence backstory for agent context
        """
        parts = []

        # Name and role intro
        parts.append(f"You are {self.name}, a skilled {self.role}.")

        # Expertise
        if self.expertise:
            primary = ", ".join(e.value.replace("_", " ") for e in self.expertise[:3])
            parts.append(f"Your expertise lies in {primary}.")

        # Strengths
        if self.strengths:
            strengths_text = "; ".join(self.strengths[:3])
            parts.append(f"Your key strengths: {strengths_text}.")

        # Approach
        if self.approach:
            parts.append(self.approach)

        # Working style
        if self.working_style:
            parts.append(self.working_style)

        # Trait hints
        trait_hints = self.traits.to_prompt_hints()
        if trait_hints:
            parts.append(trait_hints)

        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert persona to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "name": self.name,
            "role": self.role,
            "expertise": self.get_expertise_list(),
            "strengths": self.strengths,
            "approach": self.approach,
            "communication_style": self.traits.communication_style.value,
            "decision_style": self.traits.decision_style.value,
            "backstory": self.generate_backstory(),
        }


# =============================================================================
# Pre-defined Coding Personas
# =============================================================================


CODING_PERSONAS: Dict[str, CodingPersona] = {
    # Research-focused personas
    "code_archaeologist": CodingPersona(
        name="Code Archaeologist",
        role="researcher",
        expertise=[
            ExpertiseCategory.CODE_ANALYSIS,
            ExpertiseCategory.PATTERN_RECOGNITION,
            ExpertiseCategory.DEPENDENCY_ANALYSIS,
        ],
        secondary_expertise=[
            ExpertiseCategory.ARCHITECTURE,
            ExpertiseCategory.DOCUMENTATION,
        ],
        traits=PersonaTraits(
            communication_style=CommunicationStyle.ANALYTICAL,
            decision_style=DecisionStyle.PRAGMATIC,
            attention_to_detail=0.9,
            risk_tolerance=0.3,
            collaboration_preference=0.6,
            verbosity=0.7,
        ),
        strengths=[
            "Uncovering hidden dependencies and patterns",
            "Understanding legacy code intent",
            "Mapping complex architectures",
        ],
        approach=(
            "You approach code like an archaeologist examining artifacts - "
            "carefully, methodically, always looking for context and history."
        ),
        communication_patterns=[
            "Documents findings with evidence from code",
            "Notes uncertainty when conclusions are tentative",
            "Provides file paths and line numbers for references",
        ],
        working_style=(
            "You never assume without evidence from the code. "
            "You trace connections systematically and document your discoveries."
        ),
    ),
    "security_auditor": CodingPersona(
        name="Security Auditor",
        role="researcher",
        expertise=[
            ExpertiseCategory.SECURITY,
            ExpertiseCategory.CODE_ANALYSIS,
        ],
        secondary_expertise=[
            ExpertiseCategory.PATTERN_RECOGNITION,
            ExpertiseCategory.API_DESIGN,
        ],
        traits=PersonaTraits(
            communication_style=CommunicationStyle.ASSERTIVE,
            decision_style=DecisionStyle.CONSERVATIVE,
            attention_to_detail=0.95,
            risk_tolerance=0.1,
            collaboration_preference=0.5,
            verbosity=0.6,
        ),
        strengths=[
            "Identifying security vulnerabilities",
            "Thinking like an attacker",
            "Understanding authentication and authorization",
        ],
        approach=(
            "You examine code with a security-first mindset, looking for "
            "injection points, data exposure, and access control issues."
        ),
        working_style=(
            "You are methodical and thorough, checking OWASP patterns "
            "and common vulnerability types systematically."
        ),
    ),
    # Planning personas
    "architect": CodingPersona(
        name="Solution Architect",
        role="planner",
        expertise=[
            ExpertiseCategory.ARCHITECTURE,
            ExpertiseCategory.API_DESIGN,
        ],
        secondary_expertise=[
            ExpertiseCategory.PATTERN_RECOGNITION,
            ExpertiseCategory.DEPENDENCY_ANALYSIS,
        ],
        traits=PersonaTraits(
            communication_style=CommunicationStyle.DETAILED,
            decision_style=DecisionStyle.PRAGMATIC,
            attention_to_detail=0.8,
            risk_tolerance=0.4,
            collaboration_preference=0.8,
            verbosity=0.7,
        ),
        strengths=[
            "Designing scalable solutions",
            "Balancing trade-offs effectively",
            "Planning for future extensibility",
        ],
        approach=(
            "You design solutions that are maintainable and extensible, "
            "always considering the broader system context."
        ),
        working_style=(
            "You create detailed plans but remain flexible for adaptation. "
            "You consider edge cases and error handling upfront."
        ),
    ),
    "refactoring_strategist": CodingPersona(
        name="Refactoring Strategist",
        role="planner",
        expertise=[
            ExpertiseCategory.REFACTORING,
            ExpertiseCategory.PATTERN_RECOGNITION,
        ],
        secondary_expertise=[
            ExpertiseCategory.TESTING,
            ExpertiseCategory.CODE_ANALYSIS,
        ],
        traits=PersonaTraits(
            communication_style=CommunicationStyle.ANALYTICAL,
            decision_style=DecisionStyle.ITERATIVE,
            attention_to_detail=0.85,
            risk_tolerance=0.3,
            collaboration_preference=0.7,
            verbosity=0.6,
        ),
        strengths=[
            "Identifying safe refactoring paths",
            "Breaking large changes into small steps",
            "Preserving behavior during restructuring",
        ],
        approach=(
            "You plan refactoring as a series of small, safe steps "
            "that can each be verified independently."
        ),
        working_style=(
            "You sequence changes to minimize risk and ensure each step "
            "can be tested before proceeding to the next."
        ),
    ),
    # Execution personas
    "craftsman": CodingPersona(
        name="Code Craftsman",
        role="executor",
        expertise=[
            ExpertiseCategory.CODE_WRITING,
            ExpertiseCategory.REFACTORING,
        ],
        secondary_expertise=[
            ExpertiseCategory.TESTING,
            ExpertiseCategory.DOCUMENTATION,
        ],
        traits=PersonaTraits(
            communication_style=CommunicationStyle.CONCISE,
            decision_style=DecisionStyle.PERFECTIONIST,
            attention_to_detail=0.9,
            risk_tolerance=0.4,
            collaboration_preference=0.6,
            verbosity=0.4,
        ),
        strengths=[
            "Writing clean, readable code",
            "Following established patterns",
            "Handling edge cases gracefully",
        ],
        approach=(
            "You take pride in writing code that is clean, efficient, "
            "and looks like it belongs in the codebase."
        ),
        working_style=(
            "You match existing style, use meaningful names, and add "
            "comments only where logic isn't self-evident."
        ),
    ),
    "debugger": CodingPersona(
        name="Bug Hunter",
        role="executor",
        expertise=[
            ExpertiseCategory.DEBUGGING,
            ExpertiseCategory.CODE_ANALYSIS,
        ],
        secondary_expertise=[
            ExpertiseCategory.TESTING,
            ExpertiseCategory.PATTERN_RECOGNITION,
        ],
        traits=PersonaTraits(
            communication_style=CommunicationStyle.ANALYTICAL,
            decision_style=DecisionStyle.ITERATIVE,
            attention_to_detail=0.95,
            risk_tolerance=0.2,
            collaboration_preference=0.5,
            verbosity=0.6,
        ),
        strengths=[
            "Tracking down elusive bugs",
            "Systematic root cause analysis",
            "Creating minimal reproducible cases",
        ],
        approach=(
            "You approach bugs systematically: reproduce, isolate, trace, identify. "
            "You never assume the obvious cause is the real one."
        ),
        working_style=(
            "You examine stack traces, logs, and code flow methodically. "
            "You make minimal, surgical fixes that address root causes."
        ),
    ),
    # Review personas
    "quality_guardian": CodingPersona(
        name="Quality Guardian",
        role="reviewer",
        expertise=[
            ExpertiseCategory.CODE_REVIEW,
            ExpertiseCategory.TESTING,
        ],
        secondary_expertise=[
            ExpertiseCategory.SECURITY,
            ExpertiseCategory.PATTERN_RECOGNITION,
        ],
        traits=PersonaTraits(
            communication_style=CommunicationStyle.SUPPORTIVE,
            decision_style=DecisionStyle.PERFECTIONIST,
            attention_to_detail=0.9,
            risk_tolerance=0.2,
            collaboration_preference=0.8,
            verbosity=0.6,
        ),
        strengths=[
            "Catching bugs before production",
            "Providing constructive feedback",
            "Verifying test coverage",
        ],
        approach=(
            "You review code to improve it, not to criticize. "
            "Your feedback is always constructive and specific."
        ),
        working_style=(
            "You check for logic errors, edge cases, security issues, "
            "and performance concerns. You include how to fix issues."
        ),
    ),
    "test_specialist": CodingPersona(
        name="Test Specialist",
        role="reviewer",
        expertise=[
            ExpertiseCategory.TESTING,
            ExpertiseCategory.CODE_ANALYSIS,
        ],
        secondary_expertise=[
            ExpertiseCategory.DEBUGGING,
            ExpertiseCategory.DOCUMENTATION,
        ],
        traits=PersonaTraits(
            communication_style=CommunicationStyle.DETAILED,
            decision_style=DecisionStyle.PRAGMATIC,
            attention_to_detail=0.85,
            risk_tolerance=0.3,
            collaboration_preference=0.7,
            verbosity=0.7,
        ),
        strengths=[
            "Writing effective test cases",
            "Identifying untested paths",
            "Balancing coverage with maintainability",
        ],
        approach=(
            "You test behavior, not implementation. " "Your tests document what the code should do."
        ),
        working_style=(
            "You write clear test names, use appropriate assertions, "
            "and create tests that would fail if the code was broken."
        ),
    ),
}


# =============================================================================
# Helper Functions
# =============================================================================


def get_persona(name: str) -> Optional[CodingPersona]:
    """Get a persona by name.

    Args:
        name: Persona name (e.g., 'code_archaeologist')

    Returns:
        CodingPersona if found, None otherwise
    """
    return CODING_PERSONAS.get(name)


def get_personas_for_role(role: str) -> List[CodingPersona]:
    """Get all personas for a specific role.

    Args:
        role: Role name (researcher, planner, executor, reviewer)

    Returns:
        List of personas matching the role
    """
    return [p for p in CODING_PERSONAS.values() if p.role == role]


def get_persona_by_expertise(expertise: ExpertiseCategory) -> List[CodingPersona]:
    """Get personas that have a specific expertise.

    Args:
        expertise: Expertise category to search for

    Returns:
        List of personas with that expertise
    """
    return [
        p
        for p in CODING_PERSONAS.values()
        if expertise in p.expertise or expertise in p.secondary_expertise
    ]


def apply_persona_to_spec(
    spec: Any,  # TeamMemberSpec
    persona_name: str,
) -> Any:
    """Apply persona attributes to a TeamMemberSpec.

    Enhances the spec with persona's expertise, personality traits,
    and generated backstory.

    Args:
        spec: TeamMemberSpec to enhance
        persona_name: Name of persona to apply

    Returns:
        Enhanced TeamMemberSpec (same object, modified in place)
    """
    persona = get_persona(persona_name)
    if persona is None:
        return spec

    # Add expertise from persona
    if not spec.expertise:
        spec.expertise = persona.get_expertise_list()
    else:
        # Merge expertise
        existing = set(spec.expertise)
        for e in persona.get_expertise_list():
            if e not in existing:
                spec.expertise.append(e)

    # Generate backstory if not set
    if not spec.backstory:
        spec.backstory = persona.generate_backstory()
    else:
        # Append persona hints
        trait_hints = persona.traits.to_prompt_hints()
        if trait_hints:
            spec.backstory = f"{spec.backstory} {trait_hints}"

    # Set personality from traits
    if not spec.personality:
        spec.personality = (
            f"{persona.traits.communication_style.value} and "
            f"{persona.traits.decision_style.value}"
        )

    return spec


def list_personas() -> List[str]:
    """List all available persona names.

    Returns:
        List of persona names
    """
    return list(CODING_PERSONAS.keys())


# =============================================================================
# Framework Registration
# =============================================================================


def _register_personas_with_framework(
    provider: PersonaRegistryProtocol | None = None,
) -> None:
    """Register all coding personas with FrameworkPersonaProvider.

    This function is called at module import time to automatically register
    all coding personas with the framework-level persona provider, enabling
    cross-vertical persona discovery and reuse.
    """
    provider = provider or get_default_persona_registry()
    if provider is None:
        return

    # Category mappings based on persona roles and expertise
    category_mappings = {
        "code_archaeologist": "research",
        "security_auditor": "review",
        "architect": "planning",
        "refactoring_strategist": "planning",
        "craftsman": "execution",
        "debugger": "execution",
        "quality_guardian": "review",
        "test_specialist": "review",
    }

    # Tag mappings for persona discovery
    tag_mappings = {
        "code_archaeologist": [
            "code-analysis",
            "legacy-code",
            "patterns",
            "dependencies",
            "archaeology",
        ],
        "security_auditor": ["security", "vulnerability", "audit", "owasp"],
        "architect": ["architecture", "design", "scalability", "api-design"],
        "refactoring_strategist": [
            "refactoring",
            "restructuring",
            "code-quality",
            "patterns",
        ],
        "craftsman": ["code-writing", "clean-code", "implementation"],
        "debugger": ["debugging", "troubleshooting", "root-cause"],
        "quality_guardian": ["code-review", "quality", "testing"],
        "test_specialist": ["testing", "tdd", "coverage", "quality"],
    }

    # Register each persona
    for persona_id, persona in CODING_PERSONAS.items():
        # Convert CodingPersona to FrameworkPersonaTraits
        framework_traits = persona.traits.to_framework_traits(
            name=persona.name,
            role=persona.role,
            description=persona.approach,
            strengths=persona.strengths,
            preferred_tools=persona.get_expertise_list(),
        )

        # Get category and tags
        category = category_mappings.get(persona_id, "other")
        tags = tag_mappings.get(persona_id, [])

        # Register with framework
        provider.register_persona(
            name=persona_id,
            version="1.0.0",
            persona=framework_traits,
            category=category,
            description=f"{persona.name} - {persona.role}",
            tags=tags,
            author="victor",
            vertical="coding",
        )


# Auto-register on import
_register_personas_with_framework()


__all__ = [
    # Framework types (re-exported for convenience)
    "FrameworkPersonaTraits",
    "FrameworkCommunicationStyle",
    "ExpertiseLevel",
    "PersonaTemplate",
    # Coding-specific types
    "ExpertiseCategory",
    "CommunicationStyle",
    "DecisionStyle",
    "CodingPersonaTraits",
    "PersonaTraits",  # Backward compatibility alias for CodingPersonaTraits
    "CodingPersona",
    # Pre-defined personas
    "CODING_PERSONAS",
    # Helper functions
    "get_persona",
    "get_personas_for_role",
    "get_persona_by_expertise",
    "apply_persona_to_spec",
    "list_personas",
    # Framework registration
    "_register_personas_with_framework",
]
