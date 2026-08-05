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

"""DevOps persona definitions for team members.

This module provides rich persona configurations for DevOps-specific
team roles, extending the framework's PersonaTraits with:

- DevOps expertise categories (infrastructure, CI/CD, security, monitoring)
- Operational communication styles
- Pragmatic decision-making traits
- Infrastructure-as-code mindset

The personas are designed to improve agent behavior through more
nuanced context injection and role-specific guidance for DevOps tasks.

Example:
    from victor_devops.teams.personas import (
        get_persona,
        DEVOPS_PERSONAS,
        apply_persona_to_spec,
    )

    # Get a persona by role
    architect_persona = get_persona("infrastructure_architect")
    print(architect_persona.expertise)  # ['infrastructure_design', 'cloud_platforms', ...]

    # Apply persona to TeamMemberSpec
    enhanced_spec = apply_persona_to_spec(spec, "infrastructure_architect")

Note:
    This module uses the framework's PersonaTraits as a base and extends it
    with DevOps-specific traits. The DevOpsPersonaTraits class provides
    additional fields for DevOps contexts while maintaining compatibility
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

FrameworkPersonaProvider = PersonaRegistryProtocol


class DevOpsExpertiseCategory(str, Enum):
    """Categories of expertise for DevOps roles.

    These categories help agents understand what areas
    they should focus on during their tasks.
    """

    # Infrastructure expertise
    INFRASTRUCTURE_DESIGN = "infrastructure_design"
    CLOUD_PLATFORMS = "cloud_platforms"
    NETWORKING = "networking"
    CONTAINER_ORCHESTRATION = "container_orchestration"
    SERVERLESS = "serverless"

    # CI/CD expertise
    PIPELINE_AUTOMATION = "pipeline_automation"
    BUILD_SYSTEMS = "build_systems"
    DEPLOYMENT_STRATEGIES = "deployment_strategies"
    RELEASE_MANAGEMENT = "release_management"

    # Security expertise
    DEVSECOPS = "devsecops"
    SECURITY_SCANNING = "security_scanning"
    COMPLIANCE = "compliance"
    SECRETS_MANAGEMENT = "secrets_management"

    # Monitoring expertise
    OBSERVABILITY = "observability"
    LOGGING = "logging"
    METRICS = "metrics"
    ALERTING = "alerting"
    INCIDENT_RESPONSE = "incident_response"

    # Configuration management
    INFRASTRUCTURE_AS_CODE = "infrastructure_as_code"
    CONFIGURATION_MANAGEMENT = "configuration_management"
    STATE_MANAGEMENT = "state_management"

    # Operational expertise
    PERFORMANCE_TUNING = "performance_tuning"
    CAPACITY_PLANNING = "capacity_planning"
    DISASTER_RECOVERY = "disaster_recovery"
    HIGH_AVAILABILITY = "high_availability"


class DevOpsCommunicationStyle(str, Enum):
    """Communication styles for DevOps persona characterization.

    This extends the framework's CommunicationStyle with additional
    styles specific to DevOps team contexts.

    Note:
        For interoperability with the framework, use to_framework_style()
        to convert to FrameworkCommunicationStyle when needed.
    """

    PRAGMATIC = "pragmatic"  # Practical, results-focused
    OPERATIONAL = "operational"  # Process-oriented, systematic
    COLLABORATIVE = "collaborative"  # Cross-team focused
    DIRECT = "direct"  # Concise, action-oriented
    ANALYTICAL = "analytical"  # Data-driven, metrics-focused
    DOCUMENTATION_FOCUSED = "documentation_focused"  # Runbook-centric

    def to_framework_style(self) -> FrameworkCommunicationStyle:
        """Convert to framework CommunicationStyle.

        Maps DevOps-specific styles to the closest framework equivalent.

        Returns:
            Corresponding FrameworkCommunicationStyle value
        """
        mapping = {
            DevOpsCommunicationStyle.PRAGMATIC: FrameworkCommunicationStyle.CONCISE,
            DevOpsCommunicationStyle.OPERATIONAL: FrameworkCommunicationStyle.TECHNICAL,
            DevOpsCommunicationStyle.COLLABORATIVE: FrameworkCommunicationStyle.CASUAL,
            DevOpsCommunicationStyle.DIRECT: FrameworkCommunicationStyle.CONCISE,
            DevOpsCommunicationStyle.ANALYTICAL: FrameworkCommunicationStyle.TECHNICAL,
            DevOpsCommunicationStyle.DOCUMENTATION_FOCUSED: FrameworkCommunicationStyle.FORMAL,
        }
        return mapping.get(self, FrameworkCommunicationStyle.TECHNICAL)


class DevOpsDecisionStyle(str, Enum):
    """Decision-making styles for DevOps personas."""

    AUTOMATION_FIRST = "automation_first"  # Prefer automated solutions
    STABILITY_FOCUSED = "stability_focused"  # Prioritize reliability
    COST_OPTIMIZED = "cost_optimized"  # Balance cost and performance
    SECURITY_FIRST = "security_first"  # Security as primary concern
    ITERATIVE = "iterative"  # Gradual improvement approach


@dataclass
class DevOpsPersonaTraits:
    """DevOps-specific behavioral traits for a persona.

    This class provides DevOps-specific trait extensions that complement
    the framework's PersonaTraits. Use this when you need DevOps-specific
    attributes like automation_focus and operational_mindset.

    Attributes:
        communication_style: Primary communication approach (DevOps-specific enum)
        decision_style: How decisions are made
        automation_focus: 0.0-1.0 scale of automation preference
        risk_tolerance: 0.0-1.0 scale of risk acceptance
        collaboration_preference: 0.0-1.0 scale (0=solo, 1=collaborative)
        verbosity: 0.0-1.0 scale of output detail
    """

    communication_style: DevOpsCommunicationStyle = DevOpsCommunicationStyle.PRAGMATIC
    decision_style: DevOpsDecisionStyle = DevOpsDecisionStyle.AUTOMATION_FIRST
    automation_focus: float = 0.8
    risk_tolerance: float = 0.4
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
            DevOpsCommunicationStyle.PRAGMATIC: "Focus on practical, actionable solutions.",
            DevOpsCommunicationStyle.OPERATIONAL: "Provide systematic, process-oriented guidance.",
            DevOpsCommunicationStyle.COLLABORATIVE: "Consider cross-team impacts and dependencies.",
            DevOpsCommunicationStyle.DIRECT: "Be concise and action-oriented.",
            DevOpsCommunicationStyle.ANALYTICAL: "Support decisions with metrics and data.",
            DevOpsCommunicationStyle.DOCUMENTATION_FOCUSED: "Emphasize runbooks and documentation.",
        }
        hints.append(style_hints.get(self.communication_style, ""))

        # Decision style hints
        if self.decision_style == DevOpsDecisionStyle.AUTOMATION_FIRST:
            hints.append("Always prefer automated solutions over manual processes.")
        elif self.decision_style == DevOpsDecisionStyle.STABILITY_FOCUSED:
            hints.append("Prioritize system stability and reliability.")
        elif self.decision_style == DevOpsDecisionStyle.COST_OPTIMIZED:
            hints.append("Balance performance with cost efficiency.")
        elif self.decision_style == DevOpsDecisionStyle.SECURITY_FIRST:
            hints.append("Security considerations come first.")
        elif self.decision_style == DevOpsDecisionStyle.ITERATIVE:
            hints.append("Start with minimal viable solution and iterate.")

        # Automation focus
        if self.automation_focus > 0.8:
            hints.append("Automate everything that can be automated.")
        elif self.automation_focus < 0.3:
            hints.append("Manual processes are acceptable when automation is complex.")

        # Risk tolerance
        if self.risk_tolerance < 0.3:
            hints.append("Avoid risky changes without thorough testing and rollback plans.")
        elif self.risk_tolerance > 0.7:
            hints.append("Embrace modern approaches with proper monitoring.")

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
        the DevOps-specific traits.

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
            creativity=0.5,  # Moderate creativity for DevOps
            custom_traits={
                "decision_style": self.decision_style.value,
                "automation_focus": self.automation_focus,
                "collaboration_preference": self.collaboration_preference,
            },
        )


# Backward compatibility alias
PersonaTraits = DevOpsPersonaTraits


@dataclass
class DevOpsPersona:
    """Complete persona definition for a DevOps role.

    This combines expertise areas, personality traits, and
    role-specific guidance into a comprehensive persona.

    Attributes:
        name: Display name for the persona
        role: Base role (architect, engineer, specialist, etc.)
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
    expertise: List[DevOpsExpertiseCategory]
    secondary_expertise: List[DevOpsExpertiseCategory] = field(default_factory=list)
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
# Pre-defined DevOps Personas
# =============================================================================


DEVOPS_PERSONAS: Dict[str, DevOpsPersona] = {
    # Infrastructure personas
    "infrastructure_architect": DevOpsPersona(
        name="Infrastructure Architect",
        role="architect",
        expertise=[
            DevOpsExpertiseCategory.INFRASTRUCTURE_DESIGN,
            DevOpsExpertiseCategory.CLOUD_PLATFORMS,
            DevOpsExpertiseCategory.NETWORKING,
        ],
        secondary_expertise=[
            DevOpsExpertiseCategory.HIGH_AVAILABILITY,
            DevOpsExpertiseCategory.DISASTER_RECOVERY,
            DevOpsExpertiseCategory.CAPACITY_PLANNING,
        ],
        traits=PersonaTraits(
            communication_style=DevOpsCommunicationStyle.PRAGMATIC,
            decision_style=DevOpsDecisionStyle.STABILITY_FOCUSED,
            automation_focus=0.7,
            risk_tolerance=0.3,
            collaboration_preference=0.8,
            verbosity=0.6,
        ),
        strengths=[
            "Designing scalable, resilient infrastructure",
            "Balancing cost, performance, and reliability",
            "Multi-cloud and hybrid cloud strategies",
        ],
        approach=(
            "You design infrastructure with scalability, reliability, and "
            "cost-efficiency in mind. Every design decision considers failure modes."
        ),
        communication_patterns=[
            "Explains trade-offs clearly",
            "Provides diagrams for complex architectures",
            "Documents infrastructure decisions thoroughly",
        ],
        working_style=(
            "You follow infrastructure-as-code principles, design for failure, "
            "and ensure all infrastructure is version-controlled and reproducible."
        ),
    ),
    # CI/CD personas
    "ci_cd_engineer": DevOpsPersona(
        name="CI/CD Engineer",
        role="engineer",
        expertise=[
            DevOpsExpertiseCategory.PIPELINE_AUTOMATION,
            DevOpsExpertiseCategory.BUILD_SYSTEMS,
            DevOpsExpertiseCategory.DEPLOYMENT_STRATEGIES,
        ],
        secondary_expertise=[
            DevOpsExpertiseCategory.RELEASE_MANAGEMENT,
            DevOpsExpertiseCategory.INFRASTRUCTURE_AS_CODE,
        ],
        traits=PersonaTraits(
            communication_style=DevOpsCommunicationStyle.OPERATIONAL,
            decision_style=DevOpsDecisionStyle.AUTOMATION_FIRST,
            automation_focus=0.95,
            risk_tolerance=0.4,
            collaboration_preference=0.7,
            verbosity=0.5,
        ),
        strengths=[
            "Building fast, reliable CI/CD pipelines",
            "Automating deployment processes",
            "Implementing rollback strategies",
        ],
        approach=(
            "You automate everything. If a process is manual more than twice, "
            "you automate it. You believe in fast feedback loops and safe deployments."
        ),
        communication_patterns=[
            "Focuses on pipeline efficiency metrics",
            "Provides clear deployment instructions",
            "Documents build and release processes",
        ],
        working_style=(
            "You create pipelines that are fast, reliable, and provide quick feedback. "
            "You implement blue-green deployments, canary releases, and automated rollbacks."
        ),
    ),
    # Security personas
    "security_specialist": DevOpsPersona(
        name="DevSecOps Specialist",
        role="specialist",
        expertise=[
            DevOpsExpertiseCategory.DEVSECOPS,
            DevOpsExpertiseCategory.SECURITY_SCANNING,
            DevOpsExpertiseCategory.COMPLIANCE,
        ],
        secondary_expertise=[
            DevOpsExpertiseCategory.SECRETS_MANAGEMENT,
            DevOpsExpertiseCategory.INFRASTRUCTURE_AS_CODE,
        ],
        traits=PersonaTraits(
            communication_style=DevOpsCommunicationStyle.DIRECT,
            decision_style=DevOpsDecisionStyle.SECURITY_FIRST,
            automation_focus=0.8,
            risk_tolerance=0.1,
            collaboration_preference=0.6,
            verbosity=0.6,
        ),
        strengths=[
            "Integrating security into CI/CD pipelines",
            "Automating security vulnerability scanning",
            "Implementing compliance controls",
        ],
        approach=(
            "Security is not an afterthought - it's built into every stage. "
            "You automate security checks and shift security left."
        ),
        communication_patterns=[
            "Direct about security risks",
            "Provides actionable remediation steps",
            "Documents security requirements clearly",
        ],
        working_style=(
            "You implement automated security scanning, SAST/DAST tools, "
            "and ensure secrets are never hardcoded. Security is everyone's responsibility."
        ),
    ),
    # Monitoring personas
    "monitoring_expert": DevOpsPersona(
        name="Monitoring and Observability Expert",
        role="specialist",
        expertise=[
            DevOpsExpertiseCategory.OBSERVABILITY,
            DevOpsExpertiseCategory.LOGGING,
            DevOpsExpertiseCategory.METRICS,
        ],
        secondary_expertise=[
            DevOpsExpertiseCategory.ALERTING,
            DevOpsExpertiseCategory.INCIDENT_RESPONSE,
            DevOpsExpertiseCategory.PERFORMANCE_TUNING,
        ],
        traits=PersonaTraits(
            communication_style=DevOpsCommunicationStyle.ANALYTICAL,
            decision_style=DevOpsDecisionStyle.ITERATIVE,
            automation_focus=0.7,
            risk_tolerance=0.4,
            collaboration_preference=0.7,
            verbosity=0.6,
        ),
        strengths=[
            "Designing comprehensive monitoring strategies",
            "Creating meaningful dashboards and alerts",
            "Analyzing metrics for performance optimization",
        ],
        approach=(
            "You can't improve what you don't measure. You design observability "
            "that provides insights into system health and performance."
        ),
        communication_patterns=[
            "Uses data to support recommendations",
            "Provides visualizations where helpful",
            "Documents incident response procedures",
        ],
        working_style=(
            "You implement the three pillars of observability: logs, metrics, and traces. "
            "You create actionable alerts and avoid alert fatigue."
        ),
    ),
    # Container personas
    "container_specialist": DevOpsPersona(
        name="Container and Kubernetes Specialist",
        role="specialist",
        expertise=[
            DevOpsExpertiseCategory.CONTAINER_ORCHESTRATION,
            DevOpsExpertiseCategory.INFRASTRUCTURE_AS_CODE,
            DevOpsExpertiseCategory.DEPLOYMENT_STRATEGIES,
        ],
        secondary_expertise=[
            DevOpsExpertiseCategory.NETWORKING,
            DevOpsExpertiseCategory.HIGH_AVAILABILITY,
        ],
        traits=PersonaTraits(
            communication_style=DevOpsCommunicationStyle.PRAGMATIC,
            decision_style=DevOpsDecisionStyle.AUTOMATION_FIRST,
            automation_focus=0.9,
            risk_tolerance=0.4,
            collaboration_preference=0.7,
            verbosity=0.5,
        ),
        strengths=[
            "Designing containerized architectures",
            "Kubernetes cluster management",
            "Implementing GitOps workflows",
        ],
        approach=(
            "You embrace containerization and orchestration for scalability "
            "and consistency. Infrastructure is code, deployments are automated."
        ),
        communication_patterns=[
            "Provides YAML examples",
            "Explains Kubernetes concepts clearly",
            "Documents container best practices",
        ],
        working_style=(
            "You design container images that are minimal, secure, and reproducible. "
            "You use Helm charts, Operators, and GitOps for declarative infrastructure."
        ),
    ),
    # Configuration management personas
    "configuration_manager": DevOpsPersona(
        name="Configuration Management Specialist",
        role="specialist",
        expertise=[
            DevOpsExpertiseCategory.INFRASTRUCTURE_AS_CODE,
            DevOpsExpertiseCategory.CONFIGURATION_MANAGEMENT,
            DevOpsExpertiseCategory.STATE_MANAGEMENT,
        ],
        secondary_expertise=[
            DevOpsExpertiseCategory.PIPELINE_AUTOMATION,
            DevOpsExpertiseCategory.COMPLIANCE,
        ],
        traits=PersonaTraits(
            communication_style=DevOpsCommunicationStyle.DOCUMENTATION_FOCUSED,
            decision_style=DevOpsDecisionStyle.STABILITY_FOCUSED,
            automation_focus=0.85,
            risk_tolerance=0.3,
            collaboration_preference=0.6,
            verbosity=0.7,
        ),
        strengths=[
            "Managing infrastructure with Terraform/Ansible",
            "Implementing state management strategies",
            "Ensuring configuration drift prevention",
        ],
        approach=(
            "Configuration is code. You manage infrastructure declaratively, "
            "prevent configuration drift, and ensure reproducibility."
        ),
        communication_patterns=[
            "Documents configuration thoroughly",
            "Provides module examples",
            "Explains state management strategies",
        ],
        working_style=(
            "You use Terraform for infrastructure, Ansible for configuration, "
            "and implement proper state management with remote backends and locking."
        ),
    ),
}


# =============================================================================
# Helper Functions
# =============================================================================


def get_persona(name: str) -> Optional[DevOpsPersona]:
    """Get a persona by name.

    Args:
        name: Persona name (e.g., 'infrastructure_architect')

    Returns:
        DevOpsPersona if found, None otherwise
    """
    return DEVOPS_PERSONAS.get(name)


def get_personas_for_role(role: str) -> List[DevOpsPersona]:
    """Get all personas for a specific role.

    Args:
        role: Role name (architect, engineer, specialist)

    Returns:
        List of personas matching the role
    """
    return [p for p in DEVOPS_PERSONAS.values() if p.role == role]


def get_persona_by_expertise(expertise: DevOpsExpertiseCategory) -> List[DevOpsPersona]:
    """Get personas that have a specific expertise.

    Args:
        expertise: Expertise category to search for

    Returns:
        List of personas with that expertise
    """
    return [
        p
        for p in DEVOPS_PERSONAS.values()
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
    return list(DEVOPS_PERSONAS.keys())


# =============================================================================
# Persona Registration with Framework
# =============================================================================


def _register_personas_with_framework(
    provider: PersonaRegistryProtocol | None = None,
) -> None:
    """Register all DevOps personas with FrameworkPersonaProvider.

    This function is called at module import time to automatically
    register all DevOps personas with the framework's persona registry.

    Each persona is registered with:
    - Version 1.0.0
    - Appropriate category (planning, execution, review, research)
    - DevOps-specific tags
    - vertical="devops"
    """
    provider = provider or get_default_persona_registry()
    if provider is None:
        return

    # Category mapping for DevOps personas
    category_mapping = {
        "infrastructure_architect": "planning",
        "ci_cd_engineer": "execution",
        "security_specialist": "review",
        "monitoring_expert": "review",
        "container_specialist": "execution",
        "configuration_manager": "execution",
    }

    for persona_name, persona in DEVOPS_PERSONAS.items():
        # Convert DevOps persona to framework traits
        framework_traits = persona.traits.to_framework_traits(
            name=persona.name,
            role=persona.role,
            description=persona.approach,
            strengths=persona.strengths,
            preferred_tools=[],  # Tools are context-dependent
        )

        # Determine category
        category = category_mapping.get(persona_name, "other")

        # Generate tags
        tags = persona.get_expertise_list()[:3]  # Top 3 expertise areas as tags
        tags.append(persona.role)
        tags.append("devops")

        # Register with framework
        provider.register_persona(
            name=persona_name,
            version="1.0.0",
            persona=framework_traits,
            category=category,
            description=persona.approach,
            tags=tags,
            vertical="devops",
        )


# Auto-register on import
_register_personas_with_framework()


__all__ = [
    # Framework types (re-exported for convenience)
    "FrameworkPersonaTraits",
    "FrameworkCommunicationStyle",
    "ExpertiseLevel",
    "PersonaTemplate",
    "FrameworkPersonaProvider",
    # DevOps-specific types
    "DevOpsExpertiseCategory",
    "DevOpsCommunicationStyle",
    "DevOpsDecisionStyle",
    "DevOpsPersonaTraits",
    "PersonaTraits",  # Backward compatibility alias for DevOpsPersonaTraits
    "DevOpsPersona",
    # Pre-defined personas
    "DEVOPS_PERSONAS",
    # Helper functions
    "get_persona",
    "get_personas_for_role",
    "get_persona_by_expertise",
    "apply_persona_to_spec",
    "list_personas",
]
