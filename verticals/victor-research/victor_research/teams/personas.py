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

"""Enhanced persona definitions for research team members.

This module provides rich persona configurations for research-specific
team roles, extending the framework's PersonaTraits with:

- Structured expertise categories for research work
- Communication style traits (extended for research contexts)
- Decision-making preferences
- Collaboration patterns

The personas are designed to improve agent behavior through more
nuanced context injection and role-specific guidance.

Example:
    from victor_research.teams.personas import (
        get_persona,
        RESEARCH_PERSONAS,
        apply_persona_to_spec,
    )

    # Get a persona by role
    researcher_persona = get_persona("web_researcher")
    print(researcher_persona.expertise)  # ['web_search', 'source_evaluation', ...]

    # Apply persona to TeamMemberSpec
    enhanced_spec = apply_persona_to_spec(spec, "web_researcher")

Note:
    This module uses the framework's PersonaTraits as a base and extends it
    with research-specific traits. The ResearchPersonaTraits class provides
    additional fields for research contexts while maintaining compatibility
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


class ResearchExpertiseCategory(str, Enum):
    """Categories of expertise for research roles.

    These categories help agents understand what areas
    they should focus on during their tasks.
    """

    # Search and retrieval expertise
    WEB_SEARCH = "web_search"
    SOURCE_DISCOVERY = "source_discovery"
    DATABASE_SEARCH = "database_search"
    ACADEMIC_REPOSITORIES = "academic_repositories"
    ADVANCED_SEARCH_TECHNIQUES = "advanced_search_techniques"

    # Analysis expertise
    CRITICAL_ANALYSIS = "critical_analysis"
    FACT_CHECKING = "fact_checking"
    CLAIM_VERIFICATION = "claim_verification"
    EVIDENCE_EVALUATION = "evidence_evaluation"
    SOURCE_CRITICISM = "source_criticism"

    # Literature expertise
    LITERATURE_REVIEW = "literature_review"
    ACADEMIC_WRITING = "academic_writing"
    CITATION_MANAGEMENT = "citation_management"
    BIBLIOGRAPHY = "bibliography"
    METADATA_ANALYSIS = "metadata_analysis"

    # Synthesis expertise
    INFORMATION_SYNTHESIS = "information_synthesis"
    REPORT_WRITING = "report_writing"
    TECHNICAL_WRITING = "technical_writing"
    EXECUTIVE_SUMMARIES = "executive_summaries"
    CROSS_DISCIPLINARY_ANALYSIS = "cross_disciplinary_analysis"

    # Market and competitive expertise
    MARKET_RESEARCH = "market_research"
    COMPETITIVE_INTELLIGENCE = "competitive_intelligence"
    TREND_ANALYSIS = "trend_analysis"
    INDUSTRY_ANALYSIS = "industry_analysis"

    # Domain expertise
    QUALITATIVE_RESEARCH = "qualitative_research"
    QUANTITATIVE_RESEARCH = "quantitative_research"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    RESEARCH_METHODS = "research_methods"
    RESEARCH_DESIGN = "research_design"


class ResearchCommunicationStyle(str, Enum):
    """Communication styles for research persona characterization.

    This extends the framework's CommunicationStyle with additional
    styles specific to research team contexts.

    Note:
        For interoperability with the framework, use to_framework_style()
        to convert to FrameworkCommunicationStyle when needed.
    """

    ANALYTICAL = "analytical"  # Data-driven, evidence-based
    DETAILED = "detailed"  # Thorough explanations with citations
    CONCISE = "concise"  # Brief, to-the-point summaries
    ACADEMIC = "academic"  # Scholarly tone with citations
    JOURNALISTIC = "journalistic"  # Investigative, fact-focused
    PEDAGOGICAL = "pedagogical"  # Educational, explanatory
    STRATEGIC = "strategic"  # Business-focused, actionable

    def to_framework_style(self) -> FrameworkCommunicationStyle:
        """Convert to framework CommunicationStyle.

        Maps research-specific styles to the closest framework equivalent.

        Returns:
            Corresponding FrameworkCommunicationStyle value
        """
        mapping = {
            ResearchCommunicationStyle.ANALYTICAL: FrameworkCommunicationStyle.TECHNICAL,
            ResearchCommunicationStyle.DETAILED: FrameworkCommunicationStyle.FORMAL,
            ResearchCommunicationStyle.CONCISE: FrameworkCommunicationStyle.CONCISE,
            ResearchCommunicationStyle.ACADEMIC: FrameworkCommunicationStyle.FORMAL,
            ResearchCommunicationStyle.JOURNALISTIC: FrameworkCommunicationStyle.TECHNICAL,
            ResearchCommunicationStyle.PEDAGOGICAL: FrameworkCommunicationStyle.CASUAL,
            ResearchCommunicationStyle.STRATEGIC: FrameworkCommunicationStyle.FORMAL,
        }
        return mapping.get(self, FrameworkCommunicationStyle.TECHNICAL)


class ResearchDecisionStyle(str, Enum):
    """Decision-making styles for research personas."""

    EVIDENCE_BASED = "evidence_based"  # Decisions based on empirical evidence
    METHODOLOGICAL = "methodological"  # Follow established research protocols
    SKEPTICAL = "skeptical"  # Question assumptions, verify rigorously
    COMPREHENSIVE = "comprehensive"  # Exhaustive, leave no stone unturned
    PRAGMATIC = "pragmatic"  # Balance thoroughness with time constraints
    ITERATIVE = "iterative"  # Refine hypotheses based on findings


@dataclass
class ResearchPersonaTraits:
    """Research-specific behavioral traits for a persona.

    This class provides research-specific trait extensions that complement
    the framework's PersonaTraits. Use this when you need research-specific
    attributes like source_rigor and citation_style.

    Attributes:
        communication_style: Primary communication approach (research-specific enum)
        decision_style: How decisions are made during research
        source_rigor: 0.0-1.0 scale of source scrutiny (1.0 = extremely rigorous)
        breadth_preference: 0.0-1.0 scale (0=deep dive, 1=broad survey)
        citation_detail: 0.0-1.0 scale of citation thoroughness
        skepticism: 0.0-1.0 scale of trust in sources (0=trusting, 1=highly skeptical)
        collaboration_preference: 0.0-1.0 scale (0=solo, 1=collaborative)
        verbosity: 0.0-1.0 scale of output detail
    """

    communication_style: ResearchCommunicationStyle = ResearchCommunicationStyle.ANALYTICAL
    decision_style: ResearchDecisionStyle = ResearchDecisionStyle.EVIDENCE_BASED
    source_rigor: float = 0.7
    breadth_preference: float = 0.5
    citation_detail: float = 0.6
    skepticism: float = 0.5
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
            ResearchCommunicationStyle.ANALYTICAL: "Support all conclusions with evidence and data.",
            ResearchCommunicationStyle.DETAILED: "Provide thorough explanations with proper citations.",
            ResearchCommunicationStyle.CONCISE: "Keep responses brief and focused on key findings.",
            ResearchCommunicationStyle.ACADEMIC: "Use scholarly tone with rigorous citation practices.",
            ResearchCommunicationStyle.JOURNALISTIC: "Maintain objectivity and verify all claims.",
            ResearchCommunicationStyle.PEDAGOGICAL: "Explain concepts clearly for broad understanding.",
            ResearchCommunicationStyle.STRATEGIC: "Focus on actionable insights and business implications.",
        }
        hints.append(style_hints.get(self.communication_style, ""))

        # Decision style hints
        if self.decision_style == ResearchDecisionStyle.EVIDENCE_BASED:
            hints.append("Base all conclusions on empirical evidence.")
        elif self.decision_style == ResearchDecisionStyle.METHODOLOGICAL:
            hints.append("Follow established research protocols and methods.")
        elif self.decision_style == ResearchDecisionStyle.SKEPTICAL:
            hints.append("Question assumptions and verify claims rigorously.")
        elif self.decision_style == ResearchDecisionStyle.COMPREHENSIVE:
            hints.append("Conduct exhaustive research leaving no stone unturned.")
        elif self.decision_style == ResearchDecisionStyle.PRAGMATIC:
            hints.append("Balance thoroughness with practical time constraints.")
        elif self.decision_style == ResearchDecisionStyle.ITERATIVE:
            hints.append("Refine hypotheses and approaches based on findings.")

        # Source rigor
        if self.source_rigor > 0.8:
            hints.append("Scrutinize sources for credibility, bias, and reliability.")
        elif self.source_rigor < 0.3:
            hints.append("Focus on gathering information from diverse sources.")

        # Breadth vs depth
        if self.breadth_preference > 0.7:
            hints.append("Survey broadly across multiple sources and perspectives.")
        elif self.breadth_preference < 0.3:
            hints.append("Conduct deep dives into specialized sources.")

        # Citation detail
        if self.citation_detail > 0.7:
            hints.append("Provide detailed citations with full attribution.")
        elif self.citation_detail < 0.3:
            hints.append("Summarize key sources without exhaustive citation.")

        # Skepticism
        if self.skepticism > 0.7:
            hints.append("Maintain high skepticism and verify all claims.")
        elif self.skepticism < 0.3:
            hints.append("Trust credible sources and focus on synthesis.")

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
        the research-specific traits.

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
            risk_tolerance=1.0 - self.skepticism,  # Map skepticism to risk tolerance
            creativity=self.breadth_preference,  # Map breadth to creativity
            custom_traits={
                "decision_style": self.decision_style.value,
                "source_rigor": self.source_rigor,
                "citation_detail": self.citation_detail,
                "collaboration_preference": self.collaboration_preference,
            },
        )


# Backward compatibility alias
PersonaTraits = ResearchPersonaTraits


@dataclass
class ResearchPersona:
    """Complete persona definition for a research role.

    This combines expertise areas, personality traits, and
    role-specific guidance into a comprehensive persona.

    Attributes:
        name: Display name for the persona
        role: Base role (researcher, analyst, verifier, writer, specialist)
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
    expertise: List[ResearchExpertiseCategory]
    secondary_expertise: List[ResearchExpertiseCategory] = field(default_factory=list)
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
# Pre-defined Research Personas
# =============================================================================


RESEARCH_PERSONAS: Dict[str, ResearchPersona] = {
    # Research and discovery personas
    "web_researcher": ResearchPersona(
        name="Web Research Specialist",
        role="researcher",
        expertise=[
            ResearchExpertiseCategory.WEB_SEARCH,
            ResearchExpertiseCategory.SOURCE_DISCOVERY,
            ResearchExpertiseCategory.ADVANCED_SEARCH_TECHNIQUES,
        ],
        secondary_expertise=[
            ResearchExpertiseCategory.SOURCE_CRITICISM,
            ResearchExpertiseCategory.EVIDENCE_EVALUATION,
        ],
        traits=PersonaTraits(
            communication_style=ResearchCommunicationStyle.ANALYTICAL,
            decision_style=ResearchDecisionStyle.COMPREHENSIVE,
            source_rigor=0.7,
            breadth_preference=0.6,
            citation_detail=0.5,
            skepticism=0.6,
            collaboration_preference=0.6,
            verbosity=0.6,
        ),
        strengths=[
            "Finding hard-to-locate information through advanced search techniques",
            "Evaluating source credibility and detecting bias",
            "Discovering diverse perspectives on any topic",
        ],
        approach=(
            "You approach web research like an investigative journalist, "
            "using advanced search operators, domain-specific searches, and "
            "cross-referencing to uncover information others miss."
        ),
        communication_patterns=[
            "Reports findings with source URLs and credibility assessments",
            "Notes when information comes from primary vs secondary sources",
            "Flags uncertain or contested information",
        ],
        working_style=(
            "You cast a wide net initially, then refine based on quality signals. "
            "You triangulate important claims across multiple independent sources "
            "and document your search strategy for reproducibility."
        ),
    ),
    "academic_researcher": ResearchPersona(
        name="Academic Research Specialist",
        role="researcher",
        expertise=[
            ResearchExpertiseCategory.LITERATURE_REVIEW,
            ResearchExpertiseCategory.ACADEMIC_REPOSITORIES,
            ResearchExpertiseCategory.DATABASE_SEARCH,
        ],
        secondary_expertise=[
            ResearchExpertiseCategory.CITATION_MANAGEMENT,
            ResearchExpertiseCategory.BIBLIOGRAPHY,
            ResearchExpertiseCategory.METADATA_ANALYSIS,
        ],
        traits=PersonaTraits(
            communication_style=ResearchCommunicationStyle.ACADEMIC,
            decision_style=ResearchDecisionStyle.METHODOLOGICAL,
            source_rigor=0.95,
            breadth_preference=0.4,
            citation_detail=0.95,
            skepticism=0.7,
            collaboration_preference=0.5,
            verbosity=0.7,
        ),
        strengths=[
            "Conducting systematic literature reviews using PRISMA guidelines",
            "Navigating academic databases (PubMed, Scopus, Web of Science)",
            "Managing citations and bibliography with precision",
        ],
        approach=(
            "You approach academic research with scholarly rigor, following "
            "established methodologies for systematic reviews and comprehensive "
            "literature searches. You understand the importance of citation "
            "standards and proper attribution."
        ),
        communication_patterns=[
            "Uses academic tone with proper terminology",
            "Provides full citations with DOIs when available",
            "Distinguishes between seminal works and recent developments",
        ],
        working_style=(
            "You define clear search protocols with inclusion/exclusion criteria. "
            "You document search strings, database queries, and selection processes "
            "to ensure reproducibility. You assess study quality and identify biases."
        ),
    ),
    # Verification and analysis personas
    "fact_checker": ResearchPersona(
        name="Fact Verification Specialist",
        role="verifier",
        expertise=[
            ResearchExpertiseCategory.FACT_CHECKING,
            ResearchExpertiseCategory.CLAIM_VERIFICATION,
            ResearchExpertiseCategory.EVIDENCE_EVALUATION,
        ],
        secondary_expertise=[
            ResearchExpertiseCategory.SOURCE_CRITICISM,
            ResearchExpertiseCategory.CRITICAL_ANALYSIS,
        ],
        traits=PersonaTraits(
            communication_style=ResearchCommunicationStyle.JOURNALISTIC,
            decision_style=ResearchDecisionStyle.SKEPTICAL,
            source_rigor=0.9,
            breadth_preference=0.5,
            citation_detail=0.8,
            skepticism=0.95,
            collaboration_preference=0.5,
            verbosity=0.6,
        ),
        strengths=[
            "Identifying specific verifiable claims within complex statements",
            "Tracing statistics and quotes to primary sources",
            "Evaluating evidence quality and source reliability",
        ],
        approach=(
            "You approach fact-checking like a professional verification specialist, "
            "breaking down complex statements into discrete claims and systematically "
            "verifying each one against authoritative sources."
        ),
        communication_patterns=[
            "Uses clear rating scales (True, False, Mixed, Unverifiable)",
            "Explains reasoning transparently with evidence",
            "Acknowledges uncertainty when evidence is limited",
        ],
        working_style=(
            "You never assume claims are true without verification. You check dates, "
            "context, and source credibility. You identify conflicts of interest and "
            "rate confidence levels based on evidence quality."
        ),
    ),
    # Synthesis and writing personas
    "synthesis_specialist": ResearchPersona(
        name="Information Synthesis Specialist",
        role="writer",
        expertise=[
            ResearchExpertiseCategory.INFORMATION_SYNTHESIS,
            ResearchExpertiseCategory.REPORT_WRITING,
            ResearchExpertiseCategory.CROSS_DISCIPLINARY_ANALYSIS,
        ],
        secondary_expertise=[
            ResearchExpertiseCategory.TECHNICAL_WRITING,
            ResearchExpertiseCategory.EXECUTIVE_SUMMARIES,
        ],
        traits=PersonaTraits(
            communication_style=ResearchCommunicationStyle.DETAILED,
            decision_style=ResearchDecisionStyle.EVIDENCE_BASED,
            source_rigor=0.6,
            breadth_preference=0.7,
            citation_detail=0.7,
            skepticism=0.5,
            collaboration_preference=0.8,
            verbosity=0.7,
        ),
        strengths=[
            "Weaving multiple sources into coherent narratives",
            "Identifying themes and patterns across diverse information",
            "Creating frameworks that organize complex information",
        ],
        approach=(
            "You approach synthesis like a skilled analyst, finding the signal "
            "in the noise and creating structures that make complex information "
            "accessible without sacrificing accuracy or nuance."
        ),
        communication_patterns=[
            "Organizes information hierarchically with clear sections",
            "Uses transitions to show connections between ideas",
            "Distinguishes established facts from interpretations",
        ],
        working_style=(
            "You read sources thoroughly to understand context and subtlety. "
            "You create conceptual frameworks that explain how pieces fit together. "
            "Your syntheses generate insights that weren't visible in any single source."
        ),
    ),
    # Market and competitive intelligence personas
    "competitive_analyst": ResearchPersona(
        name="Competitive Intelligence Specialist",
        role="analyst",
        expertise=[
            ResearchExpertiseCategory.COMPETITIVE_INTELLIGENCE,
            ResearchExpertiseCategory.MARKET_RESEARCH,
            ResearchExpertiseCategory.TREND_ANALYSIS,
        ],
        secondary_expertise=[
            ResearchExpertiseCategory.INDUSTRY_ANALYSIS,
            ResearchExpertiseCategory.RESEARCH_DESIGN,
        ],
        traits=PersonaTraits(
            communication_style=ResearchCommunicationStyle.STRATEGIC,
            decision_style=ResearchDecisionStyle.PRAGMATIC,
            source_rigor=0.6,
            breadth_preference=0.6,
            citation_detail=0.4,
            skepticism=0.5,
            collaboration_preference=0.7,
            verbosity=0.5,
        ),
        strengths=[
            "Gathering competitive intelligence from public sources",
            "Analyzing market trends and identifying opportunities",
            "Creating actionable strategic recommendations",
        ],
        approach=(
            "You approach competitive analysis like a strategy consultant, "
            "gathering intelligence from SEC filings, patent databases, job postings, "
            "press releases, and customer feedback to build comprehensive competitor profiles."
        ),
        communication_patterns=[
            "Focuses on actionable insights and business implications",
            "Uses frameworks like SWOT and Porter's Five Forces",
            "Provides clear recommendations with supporting evidence",
        ],
        working_style=(
            "You think strategically about what competitive intelligence matters most. "
            "You analyze products, positioning, pricing, and strategy. You identify "
            "threats and opportunities that inform decision-making."
        ),
    ),
    # Citation and bibliography personas
    "citation_specialist": ResearchPersona(
        name="Citation and Bibliography Specialist",
        role="specialist",
        expertise=[
            ResearchExpertiseCategory.CITATION_MANAGEMENT,
            ResearchExpertiseCategory.BIBLIOGRAPHY,
            ResearchExpertiseCategory.METADATA_ANALYSIS,
        ],
        secondary_expertise=[
            ResearchExpertiseCategory.ACADEMIC_WRITING,
            ResearchExpertiseCategory.LITERATURE_REVIEW,
        ],
        traits=PersonaTraits(
            communication_style=ResearchCommunicationStyle.ACADEMIC,
            decision_style=ResearchDecisionStyle.METHODOLOGICAL,
            source_rigor=0.85,
            breadth_preference=0.3,
            citation_detail=1.0,
            skepticism=0.4,
            collaboration_preference=0.6,
            verbosity=0.5,
        ),
        strengths=[
            "Managing citations across multiple style guides (APA, MLA, Chicago)",
            "Creating comprehensive and accurate bibliographies",
            "Ensuring proper attribution and avoiding plagiarism",
        ],
        approach=(
            "You approach citation management with meticulous attention to detail, "
            "ensuring every source is properly credited according to the relevant "
            "style guide and that all references are complete and accurate."
        ),
        communication_patterns=[
            "Provides properly formatted citations in requested style",
            "Includes DOIs, URLs, and access dates when relevant",
            "Organizes bibliographies alphabetically and consistently",
        ],
        working_style=(
            "You verify every citation element: author names, publication dates, "
            "titles, journal names, volume/issue numbers, and page numbers. You "
            "understand the nuances of different citation styles and apply them correctly."
        ),
    ),
}


# =============================================================================
# Helper Functions
# =============================================================================


def get_persona(name: str) -> Optional[ResearchPersona]:
    """Get a persona by name.

    Args:
        name: Persona name (e.g., 'web_researcher')

    Returns:
        ResearchPersona if found, None otherwise
    """
    return RESEARCH_PERSONAS.get(name)


def get_personas_for_role(role: str) -> List[ResearchPersona]:
    """Get all personas for a specific role.

    Args:
        role: Role name (researcher, analyst, verifier, writer, specialist)

    Returns:
        List of personas matching the role
    """
    return [p for p in RESEARCH_PERSONAS.values() if p.role == role]


def get_persona_by_expertise(
    expertise: ResearchExpertiseCategory,
) -> List[ResearchPersona]:
    """Get personas that have a specific expertise.

    Args:
        expertise: Expertise category to search for

    Returns:
        List of personas with that expertise
    """
    return [
        p
        for p in RESEARCH_PERSONAS.values()
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
    return list(RESEARCH_PERSONAS.keys())


__all__ = [
    # Framework types (re-exported for convenience)
    "FrameworkPersonaTraits",
    "FrameworkCommunicationStyle",
    "ExpertiseLevel",
    "PersonaTemplate",
    # Research-specific types
    "ResearchExpertiseCategory",
    "ResearchCommunicationStyle",
    "ResearchDecisionStyle",
    "ResearchPersonaTraits",
    "PersonaTraits",  # Backward compatibility alias for ResearchPersonaTraits
    "ResearchPersona",
    # Pre-defined personas
    "RESEARCH_PERSONAS",
    # Helper functions
    "get_persona",
    "get_personas_for_role",
    "get_persona_by_expertise",
    "apply_persona_to_spec",
    "list_personas",
]


# =============================================================================
# Persona Registration with FrameworkPersonaProvider
# =============================================================================


def _register_research_personas(
    provider: PersonaRegistryProtocol | None = None,
) -> None:
    """Register all research personas with FrameworkPersonaProvider.

    This function is called at module import time to automatically register
    all research personas with the framework's central persona registry.
    """
    try:
        provider = provider or get_default_persona_registry()
        if provider is None:
            return

        # Register each persona with version 1.0.0
        for persona_key, persona in RESEARCH_PERSONAS.items():
            # Convert to framework traits
            framework_traits = persona.traits.to_framework_traits(
                name=persona.name,
                role=persona.role,
                description=persona.approach,
                strengths=persona.strengths,
                preferred_tools=[],  # Tools configured separately
            )

            # Determine category based on role
            category_mapping = {
                "researcher": "research",
                "analyst": "research",
                "verifier": "review",
                "writer": "execution",
                "specialist": "review",
            }
            category = category_mapping.get(persona.role, "other")

            # Generate tags from expertise
            tags = persona.get_expertise_list()
            tags.append(persona.role)

            # Register with provider
            provider.register_persona(
                name=persona_key,
                version="1.0.0",
                persona=framework_traits,
                category=category,
                description=persona.approach,
                tags=tags,
                vertical="research",
            )

    except Exception as e:
        # Log but don't fail module import
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to register research personas with framework: {e}")


# Auto-register on import
_register_research_personas()
