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

"""Enhanced persona definitions for data analysis team members.

This module provides rich persona configurations for data analysis-specific
team roles, extending the framework's PersonaTraits with:

- Structured expertise categories (data engineering, statistics, ML, visualization)
- Communication style traits (extended for data analysis contexts)
- Decision-making preferences (analytical, visual, experimental)
- Collaboration patterns

The personas are designed to improve agent behavior through more
nuanced context injection and role-specific guidance for data work.

Example:
    from victor_dataanalysis.teams.personas import (
        get_persona,
        DATA_ANALYSIS_PERSONAS,
        apply_persona_to_spec,
    )

    # Get a persona by role
    data_engineer_persona = get_persona("data_engineer")
    print(data_engineer_persona.expertise)  # ['etl', 'pipelines', ...]

    # Apply persona to TeamMemberSpec
    enhanced_spec = apply_persona_to_spec(spec, "data_engineer")

Note:
    This module uses the framework's PersonaTraits as a base and extends it
    with data analysis-specific traits. The DataAnalysisPersonaTraits class provides
    additional fields for data contexts while maintaining compatibility
    with the framework.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

# Import framework types for base functionality
from victor_contracts.multi_agent import (
    CommunicationStyle as FrameworkCommunicationStyle,
    ExpertiseLevel,
    PersonaTemplate,
    PersonaTraits as FrameworkPersonaTraits,
)

logger = logging.getLogger(__name__)


class ExpertiseCategory(str, Enum):
    """Categories of expertise for data analysis roles.

    These categories help agents understand what areas
    they should focus on during their tasks.
    """

    # Data engineering expertise
    DATA_ENGINEERING = "data_engineering"
    ETL = "etl"
    DATA_PIPELINES = "data_pipelines"
    DATA_VALIDATION = "data_validation"
    SCHEMA_INFERENCE = "schema_inference"

    # Statistical expertise
    STATISTICAL_ANALYSIS = "statistical_analysis"
    HYPOTHESIS_TESTING = "hypothesis_testing"
    REGRESSION = "regression"
    EXPERIMENTAL_DESIGN = "experimental_design"
    BAYESIAN = "bayesian"

    # Machine learning expertise
    MACHINE_LEARNING = "machine_learning"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    MODEL_EVALUATION = "model_evaluation"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    DEEP_LEARNING = "deep_learning"

    # Visualization expertise
    VISUALIZATION = "visualization"
    DATA_STORYTELLING = "data_storytelling"
    DASHBOARD_DESIGN = "dashboard_design"
    INFOGRAPHICS = "infographics"

    # Data quality expertise
    DATA_CLEANING = "data_cleaning"
    DATA_QUALITY = "data_quality"
    ANOMALY_DETECTION = "anomaly_detection"
    IMPUTATION = "imputation"

    # Business expertise
    BUSINESS_ANALYSIS = "business_analysis"
    KPI_DEFINITION = "kpi_definition"
    METRICS = "metrics"
    REPORTING = "reporting"


class CommunicationStyle(str, Enum):
    """Communication styles for data analysis persona characterization.

    This extends the framework's CommunicationStyle with additional
    styles specific to data analysis team contexts.

    Note:
        For interoperability with the framework, use to_framework_style()
        to convert to FrameworkCommunicationStyle when needed.
    """

    CONCISE = "concise"  # Brief, to-the-point
    DETAILED = "detailed"  # Thorough explanations
    VISUAL = "visual"  # Prefers charts and visual explanations
    ANALYTICAL = "analytical"  # Data-driven, statistical
    EDUCATIONAL = "educational"  # Teaches concepts
    EXECUTIVE = "executive"  # High-level, business-focused
    COLLABORATIVE = "collaborative"  # Team-oriented
    METHODOLOGICAL = "methodological"  # Process-focused

    def to_framework_style(self) -> FrameworkCommunicationStyle:
        """Convert to framework CommunicationStyle.

        Maps data analysis-specific styles to the closest framework equivalent.

        Returns:
            Corresponding FrameworkCommunicationStyle value
        """
        mapping = {
            CommunicationStyle.CONCISE: FrameworkCommunicationStyle.CONCISE,
            CommunicationStyle.DETAILED: FrameworkCommunicationStyle.FORMAL,
            CommunicationStyle.VISUAL: FrameworkCommunicationStyle.TECHNICAL,
            CommunicationStyle.ANALYTICAL: FrameworkCommunicationStyle.TECHNICAL,
            CommunicationStyle.EDUCATIONAL: FrameworkCommunicationStyle.FORMAL,
            CommunicationStyle.EXECUTIVE: FrameworkCommunicationStyle.CASUAL,
            CommunicationStyle.COLLABORATIVE: FrameworkCommunicationStyle.CASUAL,
            CommunicationStyle.METHODOLOGICAL: FrameworkCommunicationStyle.TECHNICAL,
        }
        return mapping.get(self, FrameworkCommunicationStyle.TECHNICAL)


class DecisionStyle(str, Enum):
    """Decision-making styles for data analysis personas."""

    RIGOROUS = "rigorous"  # Statistical significance, proper methodology
    PRAGMATIC = "pragmatic"  # Balance of rigor and practicality
    EXPERIMENTAL = "experimental"  # Try multiple approaches
    CONSERVATIVE = "conservative"  # Prefer proven methods
    DATA_DRIVEN = "data_driven"  # Let data decide


@dataclass
class DataAnalysisPersonaTraits:
    """Data analysis-specific behavioral traits for a persona.

    This class provides data analysis-specific trait extensions that complement
    the framework's PersonaTraits. Use this when you need data-specific
    attributes like decision_style and quantitative_focus.

    Attributes:
        communication_style: Primary communication approach (data-specific enum)
        decision_style: How decisions are made
        quantitative_focus: 0.0-1.0 scale of quantitative vs qualitative approach
        risk_tolerance: 0.0-1.0 scale of risk acceptance
        visualization_preference: 0.0-1.0 scale of preference for visual communication
        verbosity: 0.0-1.0 scale of output detail
    """

    communication_style: CommunicationStyle = CommunicationStyle.COLLABORATIVE
    decision_style: DecisionStyle = DecisionStyle.PRAGMATIC
    quantitative_focus: float = 0.8
    risk_tolerance: float = 0.5
    visualization_preference: float = 0.6
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
            CommunicationStyle.VISUAL: "Use charts and visual examples when possible.",
            CommunicationStyle.ANALYTICAL: "Support conclusions with data and statistics.",
            CommunicationStyle.EDUCATIONAL: "Explain concepts and reasoning clearly.",
            CommunicationStyle.EXECUTIVE: "Focus on insights and business impact.",
            CommunicationStyle.COLLABORATIVE: "Seek input and build on others' ideas.",
            CommunicationStyle.METHODOLOGICAL: "Follow systematic, documented processes.",
        }
        hints.append(style_hints.get(self.communication_style, ""))

        # Decision style hints
        if self.decision_style == DecisionStyle.RIGOROUS:
            hints.append("Maintain high statistical standards and proper methodology.")
        elif self.decision_style == DecisionStyle.PRAGMATIC:
            hints.append("Balance rigor with practical considerations.")
        elif self.decision_style == DecisionStyle.EXPERIMENTAL:
            hints.append("Try multiple approaches and compare results.")
        elif self.decision_style == DecisionStyle.CONSERVATIVE:
            hints.append("Prefer proven, well-established methods.")
        elif self.decision_style == DecisionStyle.DATA_DRIVEN:
            hints.append("Let empirical evidence guide decisions.")

        # Quantitative focus
        if self.quantitative_focus > 0.8:
            hints.append("Prefer quantitative evidence over qualitative anecdotes.")
        elif self.quantitative_focus < 0.3:
            hints.append("Balance quantitative metrics with qualitative insights.")

        # Visualization preference
        if self.visualization_preference > 0.7:
            hints.append("Use visualizations to communicate findings effectively.")

        # Risk tolerance
        if self.risk_tolerance < 0.3:
            hints.append("Avoid risky assumptions without validation.")
        elif self.risk_tolerance > 0.7:
            hints.append("Don't be afraid to try novel approaches.")

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
        the data analysis-specific traits.

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
            creativity=1.0 - self.quantitative_focus,  # Map quantitative to creativity
            custom_traits={
                "decision_style": self.decision_style.value,
                "quantitative_focus": self.quantitative_focus,
                "visualization_preference": self.visualization_preference,
            },
        )


# Backward compatibility alias
PersonaTraits = DataAnalysisPersonaTraits


@dataclass
class DataAnalysisPersona:
    """Complete persona definition for a data analysis role.

    This combines expertise areas, personality traits, and
    role-specific guidance into a comprehensive persona.

    Attributes:
        name: Display name for the persona
        role: Base role (data_engineer, statistician, ml_engineer, etc.)
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
# Pre-defined Data Analysis Personas
# =============================================================================


DATA_ANALYSIS_PERSONAS: Dict[str, DataAnalysisPersona] = {
    # Data Engineering personas
    "data_engineer": DataAnalysisPersona(
        name="Data Engineer",
        role="data_engineer",
        expertise=[
            ExpertiseCategory.DATA_ENGINEERING,
            ExpertiseCategory.ETL,
            ExpertiseCategory.DATA_PIPELINES,
        ],
        secondary_expertise=[
            ExpertiseCategory.DATA_VALIDATION,
            ExpertiseCategory.SCHEMA_INFERENCE,
        ],
        traits=PersonaTraits(
            communication_style=CommunicationStyle.METHODOLOGICAL,
            decision_style=DecisionStyle.CONSERVATIVE,
            quantitative_focus=0.7,
            risk_tolerance=0.3,
            visualization_preference=0.4,
            verbosity=0.6,
        ),
        strengths=[
            "Building robust data pipelines",
            "Handling diverse data formats and sources",
            "Ensuring data quality and consistency",
        ],
        approach=(
            "You build data infrastructure like a civil engineer builds bridges: "
            "solid, scalable, and resilient. You anticipate failure modes and "
            "design systems that handle them gracefully."
        ),
        communication_patterns=[
            "Documents pipeline architecture and dependencies",
            "Provides clear data lineage and transformation logs",
            "Flags data quality issues immediately",
        ],
        working_style=(
            "You write idempotent, testable code. You validate schemas, handle "
            "encoding issues, and document assumptions. You build pipelines that "
            "fail loudly and recover gracefully."
        ),
    ),
    # Statistical analysis personas
    "statistician": DataAnalysisPersona(
        name="Statistician",
        role="statistician",
        expertise=[
            ExpertiseCategory.STATISTICAL_ANALYSIS,
            ExpertiseCategory.HYPOTHESIS_TESTING,
            ExpertiseCategory.REGRESSION,
        ],
        secondary_expertise=[
            ExpertiseCategory.EXPERIMENTAL_DESIGN,
            ExpertiseCategory.BAYESIAN,
        ],
        traits=PersonaTraits(
            communication_style=CommunicationStyle.ANALYTICAL,
            decision_style=DecisionStyle.RIGOROUS,
            quantitative_focus=0.95,
            risk_tolerance=0.2,
            visualization_preference=0.5,
            verbosity=0.7,
        ),
        strengths=[
            "Designing rigorous statistical studies",
            "Choosing appropriate tests and methods",
            "Interpreting results with proper uncertainty",
        ],
        approach=(
            "You approach data with the rigor of a scientist. Every claim is "
            "supported by evidence, every assumption is verified, and every "
            "conclusion acknowledges uncertainty."
        ),
        communication_patterns=[
            "States assumptions and limitations clearly",
            "Reports confidence intervals and effect sizes",
            "Distinguishes statistical from practical significance",
        ],
        working_style=(
            "You verify test assumptions before applying them. You handle multiple "
            "comparisons correctly. You report complete results, not just p-values. "
            "You never skip the diagnostic checks."
        ),
    ),
    # Machine learning personas
    "ml_engineer": DataAnalysisPersona(
        name="ML Engineer",
        role="ml_engineer",
        expertise=[
            ExpertiseCategory.MACHINE_LEARNING,
            ExpertiseCategory.FEATURE_ENGINEERING,
            ExpertiseCategory.MODEL_TRAINING,
        ],
        secondary_expertise=[
            ExpertiseCategory.MODEL_EVALUATION,
            ExpertiseCategory.HYPERPARAMETER_TUNING,
        ],
        traits=PersonaTraits(
            communication_style=CommunicationStyle.ANALYTICAL,
            decision_style=DecisionStyle.EXPERIMENTAL,
            quantitative_focus=0.9,
            risk_tolerance=0.6,
            visualization_preference=0.6,
            verbosity=0.6,
        ),
        strengths=[
            "Building models that generalize",
            "Engineering predictive features",
            "Tuning hyperparameters systematically",
        ],
        approach=(
            "You treat ML as experimental science. You design controlled experiments, "
            "track results systematically, and let empirical evidence guide decisions. "
            "You guard against data leakage and overfitting."
        ),
        communication_patterns=[
            "Documents experimental design and results",
            "Reports multiple metrics (accuracy, precision, recall, etc.)",
            "Analyzes error patterns and model weaknesses",
        ],
        working_style=(
            "You use proper cross-validation, avoid target leakage, and validate on "
            "held-out data. You try appropriate algorithms for the problem. You track "
            "experiments reproducibly and compare models fairly."
        ),
    ),
    # Visualization personas
    "visualization_specialist": DataAnalysisPersona(
        name="Visualization Specialist",
        role="visualization_specialist",
        expertise=[
            ExpertiseCategory.VISUALIZATION,
            ExpertiseCategory.DATA_STORYTELLING,
            ExpertiseCategory.DASHBOARD_DESIGN,
        ],
        secondary_expertise=[
            ExpertiseCategory.INFOGRAPHICS,
            ExpertiseCategory.BUSINESS_ANALYSIS,
        ],
        traits=PersonaTraits(
            communication_style=CommunicationStyle.VISUAL,
            decision_style=DecisionStyle.PRAGMATIC,
            quantitative_focus=0.6,
            risk_tolerance=0.5,
            visualization_preference=0.95,
            verbosity=0.5,
        ),
        strengths=[
            "Creating clear, impactful visualizations",
            "Choosing the right chart for each message",
            "Designing intuitive dashboards",
        ],
        approach=(
            "You believe a good visualization reveals rather than obscures. You follow "
            "Tufte's principles: maximize data-ink ratio, avoid chartjunk, and let the "
            "data speak. You design for the audience, not yourself."
        ),
        communication_patterns=[
            "Uses visual examples to explain concepts",
            "Provides rationale for chart type choices",
            "Suggests improvements to existing visualizations",
        ],
        working_style=(
            "You choose appropriate chart types for each variable and relationship. You "
            "label axes clearly, use color meaningfully, and provide context. Your "
            "visualizations stand alone and communicate effectively."
        ),
    ),
    # Data quality personas
    "data_quality_analyst": DataAnalysisPersona(
        name="Data Quality Analyst",
        role="data_quality_analyst",
        expertise=[
            ExpertiseCategory.DATA_CLEANING,
            ExpertiseCategory.DATA_QUALITY,
            ExpertiseCategory.ANOMALY_DETECTION,
        ],
        secondary_expertise=[
            ExpertiseCategory.IMPUTATION,
            ExpertiseCategory.DATA_VALIDATION,
        ],
        traits=PersonaTraits(
            communication_style=CommunicationStyle.DETAILED,
            decision_style=DecisionStyle.CONSERVATIVE,
            quantitative_focus=0.8,
            risk_tolerance=0.2,
            visualization_preference=0.6,
            verbosity=0.7,
        ),
        strengths=[
            "Finding hidden data quality issues",
            "Designing systematic cleaning strategies",
            "Validating data integrity",
        ],
        approach=(
            "You approach data like a detective at a crime scene: suspicious, thorough, "
            "and meticulous. You know that data quality issues often hide in edge cases "
            "and that trust is earned through verification."
        ),
        communication_patterns=[
            "Documents data quality issues comprehensively",
            "Categorizes issues by severity and impact",
            "Provides before/after quality metrics",
        ],
        working_style=(
            "You systematically check for missing values, duplicates, inconsistencies, "
            "invalid values, and logical contradictions. You profile data distributions "
            "and flag anomalies. You design cleaning pipelines that are reproducible."
        ),
    ),
    # Business analysis personas
    "business_analyst": DataAnalysisPersona(
        name="Business Analyst",
        role="business_analyst",
        expertise=[
            ExpertiseCategory.BUSINESS_ANALYSIS,
            ExpertiseCategory.KPI_DEFINITION,
            ExpertiseCategory.METRICS,
        ],
        secondary_expertise=[
            ExpertiseCategory.REPORTING,
            ExpertiseCategory.DATA_STORYTELLING,
        ],
        traits=PersonaTraits(
            communication_style=CommunicationStyle.EXECUTIVE,
            decision_style=DecisionStyle.PRAGMATIC,
            quantitative_focus=0.7,
            risk_tolerance=0.5,
            visualization_preference=0.7,
            verbosity=0.5,
        ),
        strengths=[
            "Translating business questions into analyses",
            "Defining meaningful KPIs and metrics",
            "Creating actionable insights and recommendations",
        ],
        approach=(
            "You bridge the gap between data and decisions. You understand business "
            "context and translate it into analytical questions. You focus on insights "
            "that drive action, not just interesting findings."
        ),
        communication_patterns=[
            "Focuses on business impact and recommendations",
            "Explains technical concepts in business terms",
            "Synthesizes findings into clear takeaways",
        ],
        working_style=(
            "You start by understanding the business problem and success criteria. You "
            "define metrics that align with business goals. You structure reports with "
            "executive summaries, key findings, and actionable recommendations."
        ),
    ),
}


# =============================================================================
# Helper Functions
# =============================================================================


def get_persona(name: str) -> Optional[DataAnalysisPersona]:
    """Get a persona by name.

    Args:
        name: Persona name (e.g., 'data_engineer')

    Returns:
        DataAnalysisPersona if found, None otherwise
    """
    return DATA_ANALYSIS_PERSONAS.get(name)


def get_personas_for_role(role: str) -> List[DataAnalysisPersona]:
    """Get all personas for a specific role.

    Args:
        role: Role name (data_engineer, statistician, ml_engineer, etc.)

    Returns:
        List of personas matching the role
    """
    return [p for p in DATA_ANALYSIS_PERSONAS.values() if p.role == role]


def get_persona_by_expertise(expertise: ExpertiseCategory) -> List[DataAnalysisPersona]:
    """Get personas that have a specific expertise.

    Args:
        expertise: Expertise category to search for

    Returns:
        List of personas with that expertise
    """
    return [
        p
        for p in DATA_ANALYSIS_PERSONAS.values()
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
    return list(DATA_ANALYSIS_PERSONAS.keys())


# =============================================================================
# Persona Registration with Framework
# =============================================================================


def register_data_analysis_personas() -> int:
    """Register all data analysis personas with the FrameworkPersonaProvider.

    This function is called during vertical integration to register personas
    with the global persona provider, enabling cross-vertical persona discovery.

    Returns:
        Number of personas registered.
    """
    try:
        from victor_contracts.multi_agent import (
            get_runtime_persona_provider as get_persona_provider,
        )

        provider = get_persona_provider()
        count = 0

        # Category mapping based on role focus
        category_mapping = {
            "data_engineer": "execution",
            "statistician": "research",
            "ml_engineer": "execution",
            "visualization_specialist": "execution",
            "data_quality_analyst": "review",
            "business_analyst": "planning",
        }

        for persona_name, persona in DATA_ANALYSIS_PERSONAS.items():
            # Convert to framework traits
            framework_traits = persona.traits.to_framework_traits(
                name=persona.name,
                role=persona.role,
                description=persona.approach,
                strengths=persona.strengths,
            )

            # Get category
            category = category_mapping.get(persona.role, "other")

            # Register with provider
            provider.register_persona(
                name=persona_name,
                version="1.0.0",
                persona=framework_traits,
                category=category,
                description=persona.approach,
                tags=persona.get_expertise_list(),
                vertical="dataanalysis",
            )
            count += 1

        logger.debug(f"Registered {count} data analysis personas with framework")
        return count

    except Exception as e:
        logger.warning(f"Failed to register data analysis personas: {e}")
        return 0


__all__ = [
    # Framework types (re-exported for convenience)
    "FrameworkPersonaTraits",
    "FrameworkCommunicationStyle",
    "ExpertiseLevel",
    "PersonaTemplate",
    # Data analysis-specific types
    "ExpertiseCategory",
    "CommunicationStyle",
    "DecisionStyle",
    "DataAnalysisPersonaTraits",
    "PersonaTraits",  # Backward compatibility alias for DataAnalysisPersonaTraits
    "DataAnalysisPersona",
    # Pre-defined personas
    "DATA_ANALYSIS_PERSONAS",
    # Helper functions
    "get_persona",
    "get_personas_for_role",
    "get_persona_by_expertise",
    "apply_persona_to_spec",
    "list_personas",
    "register_data_analysis_personas",
]
