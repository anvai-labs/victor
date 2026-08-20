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

"""Tests for victor.framework.multi_agent.personas module.

These tests verify the PersonaTraits and PersonaTemplate classes
that provide generic persona definitions for multi-agent systems.
"""

import pytest

from victor.framework.multi_agent.personas import (
    CommunicationStyle,
    ExpertiseLevel,
    PersonaTemplate,
    PersonaTraits,
)

# =============================================================================
# CommunicationStyle Enum Tests
# =============================================================================


class TestCommunicationStyle:
    """Tests for CommunicationStyle enum."""

    def test_formal_value(self):
        """FORMAL should have correct string value."""
        assert CommunicationStyle.FORMAL.value == "formal"

    def test_casual_value(self):
        """CASUAL should have correct string value."""
        assert CommunicationStyle.CASUAL.value == "casual"

    def test_technical_value(self):
        """TECHNICAL should have correct string value."""
        assert CommunicationStyle.TECHNICAL.value == "technical"

    def test_concise_value(self):
        """CONCISE should have correct string value."""
        assert CommunicationStyle.CONCISE.value == "concise"

    def test_all_values_unique(self):
        """All enum values should be unique."""
        values = [s.value for s in CommunicationStyle]
        assert len(values) == len(set(values))


# =============================================================================
# ExpertiseLevel Enum Tests
# =============================================================================


class TestExpertiseLevel:
    """Tests for ExpertiseLevel enum."""

    def test_novice_value(self):
        """NOVICE should have correct string value."""
        assert ExpertiseLevel.NOVICE.value == "novice"

    def test_intermediate_value(self):
        """INTERMEDIATE should have correct string value."""
        assert ExpertiseLevel.INTERMEDIATE.value == "intermediate"

    def test_expert_value(self):
        """EXPERT should have correct string value."""
        assert ExpertiseLevel.EXPERT.value == "expert"

    def test_specialist_value(self):
        """SPECIALIST should have correct string value."""
        assert ExpertiseLevel.SPECIALIST.value == "specialist"

    def test_all_values_unique(self):
        """All enum values should be unique."""
        values = [l.value for l in ExpertiseLevel]
        assert len(values) == len(set(values))


# =============================================================================
# PersonaTraits Creation Tests
# =============================================================================


class TestPersonaTraitsCreation:
    """Tests for PersonaTraits dataclass creation."""

    def test_required_fields(self):
        """PersonaTraits should require name, role, and description."""
        persona = PersonaTraits(
            name="Test Agent",
            role="tester",
            description="A test agent for unit tests.",
        )

        assert persona.name == "Test Agent"
        assert persona.role == "tester"
        assert persona.description == "A test agent for unit tests."

    def test_default_communication_style(self):
        """PersonaTraits should default to TECHNICAL communication style."""
        persona = PersonaTraits(
            name="Agent",
            role="worker",
            description="Description",
        )

        assert persona.communication_style == CommunicationStyle.TECHNICAL

    def test_default_expertise_level(self):
        """PersonaTraits should default to EXPERT expertise level."""
        persona = PersonaTraits(
            name="Agent",
            role="worker",
            description="Description",
        )

        assert persona.expertise_level == ExpertiseLevel.EXPERT

    def test_default_verbosity(self):
        """PersonaTraits should default to 0.5 verbosity."""
        persona = PersonaTraits(
            name="Agent",
            role="worker",
            description="Description",
        )

        assert persona.verbosity == 0.5

    def test_default_lists_empty(self):
        """PersonaTraits should have empty lists by default."""
        persona = PersonaTraits(
            name="Agent",
            role="worker",
            description="Description",
        )

        assert persona.strengths == []
        assert persona.weaknesses == []
        assert persona.preferred_tools == []

    def test_default_risk_tolerance(self):
        """PersonaTraits should default to 0.5 risk tolerance."""
        persona = PersonaTraits(
            name="Agent",
            role="worker",
            description="Description",
        )

        assert persona.risk_tolerance == 0.5

    def test_default_creativity(self):
        """PersonaTraits should default to 0.5 creativity."""
        persona = PersonaTraits(
            name="Agent",
            role="worker",
            description="Description",
        )

        assert persona.creativity == 0.5

    def test_default_custom_traits(self):
        """PersonaTraits should have empty custom_traits by default."""
        persona = PersonaTraits(
            name="Agent",
            role="worker",
            description="Description",
        )

        assert persona.custom_traits == {}

    def test_all_fields(self):
        """PersonaTraits should accept all fields together."""
        persona = PersonaTraits(
            name="Full Agent",
            role="analyst",
            description="Full description",
            communication_style=CommunicationStyle.FORMAL,
            expertise_level=ExpertiseLevel.SPECIALIST,
            verbosity=0.8,
            strengths=["analysis", "documentation"],
            weaknesses=["speed"],
            preferred_tools=["read_file", "grep"],
            risk_tolerance=0.3,
            creativity=0.7,
            custom_traits={"domain": "security"},
        )

        assert persona.name == "Full Agent"
        assert persona.communication_style == CommunicationStyle.FORMAL
        assert persona.expertise_level == ExpertiseLevel.SPECIALIST
        assert persona.verbosity == 0.8
        assert persona.strengths == ["analysis", "documentation"]
        assert persona.weaknesses == ["speed"]
        assert persona.preferred_tools == ["read", "grep"]
        assert persona.risk_tolerance == 0.3
        assert persona.creativity == 0.7
        assert persona.custom_traits == {"domain": "security"}


# =============================================================================
# PersonaTraits Validation Tests
# =============================================================================


class TestPersonaTraitsValidation:
    """Tests for PersonaTraits validation."""

    def test_verbosity_below_zero_raises(self):
        """PersonaTraits should raise for verbosity < 0."""
        with pytest.raises(ValueError, match="verbosity"):
            PersonaTraits(
                name="Agent",
                role="worker",
                description="Desc",
                verbosity=-0.1,
            )

    def test_verbosity_above_one_raises(self):
        """PersonaTraits should raise for verbosity > 1."""
        with pytest.raises(ValueError, match="verbosity"):
            PersonaTraits(
                name="Agent",
                role="worker",
                description="Desc",
                verbosity=1.1,
            )

    def test_risk_tolerance_below_zero_raises(self):
        """PersonaTraits should raise for risk_tolerance < 0."""
        with pytest.raises(ValueError, match="risk_tolerance"):
            PersonaTraits(
                name="Agent",
                role="worker",
                description="Desc",
                risk_tolerance=-0.5,
            )

    def test_risk_tolerance_above_one_raises(self):
        """PersonaTraits should raise for risk_tolerance > 1."""
        with pytest.raises(ValueError, match="risk_tolerance"):
            PersonaTraits(
                name="Agent",
                role="worker",
                description="Desc",
                risk_tolerance=2.0,
            )

    def test_creativity_below_zero_raises(self):
        """PersonaTraits should raise for creativity < 0."""
        with pytest.raises(ValueError, match="creativity"):
            PersonaTraits(
                name="Agent",
                role="worker",
                description="Desc",
                creativity=-0.1,
            )

    def test_creativity_above_one_raises(self):
        """PersonaTraits should raise for creativity > 1."""
        with pytest.raises(ValueError, match="creativity"):
            PersonaTraits(
                name="Agent",
                role="worker",
                description="Desc",
                creativity=1.5,
            )

    def test_boundary_values_valid(self):
        """PersonaTraits should accept boundary values 0.0 and 1.0."""
        persona = PersonaTraits(
            name="Agent",
            role="worker",
            description="Desc",
            verbosity=0.0,
            risk_tolerance=1.0,
            creativity=0.0,
        )

        assert persona.verbosity == 0.0
        assert persona.risk_tolerance == 1.0
        assert persona.creativity == 0.0


# =============================================================================
# PersonaTraits to_system_prompt_fragment Tests
# =============================================================================


class TestPersonaTraitsSystemPrompt:
    """Tests for PersonaTraits.to_system_prompt_fragment() method."""

    def test_includes_name_and_role(self):
        """to_system_prompt_fragment should include name and role."""
        persona = PersonaTraits(
            name="Code Reviewer",
            role="reviewer",
            description="Reviews code for quality.",
        )

        fragment = persona.to_system_prompt_fragment()

        assert "Code Reviewer" in fragment
        assert "reviewer" in fragment

    def test_includes_description(self):
        """to_system_prompt_fragment should include description."""
        persona = PersonaTraits(
            name="Agent",
            role="worker",
            description="Detailed description of the agent's purpose.",
        )

        fragment = persona.to_system_prompt_fragment()

        assert "Detailed description" in fragment

    def test_includes_communication_style(self):
        """to_system_prompt_fragment should include communication style."""
        persona = PersonaTraits(
            name="Agent",
            role="worker",
            description="Desc",
            communication_style=CommunicationStyle.FORMAL,
        )

        fragment = persona.to_system_prompt_fragment()

        assert "formal" in fragment.lower()

    def test_includes_strengths(self):
        """to_system_prompt_fragment should include strengths."""
        persona = PersonaTraits(
            name="Agent",
            role="worker",
            description="Desc",
            strengths=["analysis", "problem solving"],
        )

        fragment = persona.to_system_prompt_fragment()

        assert "analysis" in fragment
        assert "problem solving" in fragment

    def test_no_strengths_line_when_empty(self):
        """to_system_prompt_fragment should not include Strengths when empty."""
        persona = PersonaTraits(
            name="Agent",
            role="worker",
            description="Desc",
            strengths=[],
        )

        fragment = persona.to_system_prompt_fragment()

        assert "Strengths:" not in fragment


# =============================================================================
# PersonaTraits Serialization Tests
# =============================================================================


class TestPersonaTraitsSerialization:
    """Tests for PersonaTraits serialization methods."""

    def test_to_dict_includes_all_fields(self):
        """to_dict should include all persona trait fields."""
        persona = PersonaTraits(
            name="Agent",
            role="worker",
            description="Description",
            communication_style=CommunicationStyle.CONCISE,
            expertise_level=ExpertiseLevel.INTERMEDIATE,
            verbosity=0.3,
            strengths=["quick"],
            weaknesses=["thorough"],
            preferred_tools=["grep"],
            risk_tolerance=0.6,
            creativity=0.4,
            custom_traits={"key": "value"},
        )

        data = persona.to_dict()

        assert data["name"] == "Agent"
        assert data["role"] == "worker"
        assert data["description"] == "Description"
        assert data["communication_style"] == "concise"
        assert data["expertise_level"] == "intermediate"
        assert data["verbosity"] == 0.3
        assert data["strengths"] == ["quick"]
        assert data["weaknesses"] == ["thorough"]
        assert data["preferred_tools"] == ["grep"]
        assert data["risk_tolerance"] == 0.6
        assert data["creativity"] == 0.4
        assert data["custom_traits"] == {"key": "value"}

    def test_from_dict_creates_persona(self):
        """from_dict should create PersonaTraits from dictionary."""
        data = {
            "name": "Restored Agent",
            "role": "restorer",
            "description": "Restored from dict",
            "communication_style": "formal",
            "expertise_level": "specialist",
            "verbosity": 0.7,
            "strengths": ["restoration"],
            "weaknesses": [],
            "preferred_tools": [],
            "risk_tolerance": 0.5,
            "creativity": 0.5,
            "custom_traits": {},
        }

        persona = PersonaTraits.from_dict(data)

        assert persona.name == "Restored Agent"
        assert persona.role == "restorer"
        assert persona.communication_style == CommunicationStyle.FORMAL
        assert persona.expertise_level == ExpertiseLevel.SPECIALIST
        assert persona.verbosity == 0.7

    def test_round_trip_serialization(self):
        """to_dict and from_dict should round-trip correctly."""
        original = PersonaTraits(
            name="Round Trip",
            role="tester",
            description="Testing round trip",
            communication_style=CommunicationStyle.TECHNICAL,
            expertise_level=ExpertiseLevel.EXPERT,
            strengths=["testing"],
        )

        data = original.to_dict()
        restored = PersonaTraits.from_dict(data)

        assert restored.name == original.name
        assert restored.role == original.role
        assert restored.communication_style == original.communication_style
        assert restored.expertise_level == original.expertise_level
        assert restored.strengths == original.strengths


# =============================================================================
# PersonaTemplate Tests
# =============================================================================


class TestPersonaTemplate:
    """Tests for PersonaTemplate class."""

    @pytest.fixture
    def base_persona(self):
        """Create a base persona for template testing."""
        return PersonaTraits(
            name="Base Agent",
            role="base_role",
            description="Base description",
            communication_style=CommunicationStyle.TECHNICAL,
            expertise_level=ExpertiseLevel.EXPERT,
        )

    def test_create_without_overrides(self, base_persona):
        """create should return persona with base traits when no overrides."""
        template = PersonaTemplate(base_traits=base_persona)

        persona = template.create()

        assert persona.name == "Base Agent"
        assert persona.role == "base_role"
        assert persona.communication_style == CommunicationStyle.TECHNICAL

    def test_create_with_template_overrides(self, base_persona):
        """create should apply template overrides."""
        template = PersonaTemplate(
            base_traits=base_persona,
            overrides={"name": "Override Agent", "verbosity": 0.8},
        )

        persona = template.create()

        assert persona.name == "Override Agent"
        assert persona.verbosity == 0.8
        assert persona.role == "base_role"  # Not overridden

    def test_create_with_kwargs_overrides(self, base_persona):
        """create should apply kwargs overrides."""
        template = PersonaTemplate(base_traits=base_persona)

        persona = template.create(name="Kwargs Agent", creativity=0.9)

        assert persona.name == "Kwargs Agent"
        assert persona.creativity == 0.9

    def test_kwargs_override_template_overrides(self, base_persona):
        """kwargs should override template overrides."""
        template = PersonaTemplate(
            base_traits=base_persona,
            overrides={"name": "Template Name"},
        )

        persona = template.create(name="Kwargs Name")

        assert persona.name == "Kwargs Name"

    def test_create_preserves_enum_types(self, base_persona):
        """create should preserve enum types correctly."""
        template = PersonaTemplate(base_traits=base_persona)

        persona = template.create(
            communication_style=CommunicationStyle.FORMAL,
            expertise_level=ExpertiseLevel.NOVICE,
        )

        assert persona.communication_style == CommunicationStyle.FORMAL
        assert persona.expertise_level == ExpertiseLevel.NOVICE

    def test_create_multiple_personas(self, base_persona):
        """create should produce independent persona instances."""
        template = PersonaTemplate(base_traits=base_persona)

        persona1 = template.create(name="Agent 1")
        persona2 = template.create(name="Agent 2")

        assert persona1.name == "Agent 1"
        assert persona2.name == "Agent 2"
        # Modify one should not affect the other
        persona1.strengths.append("test")
        assert "test" not in persona2.strengths


# =============================================================================
# Module Export Tests
# =============================================================================


class TestPersonaExports:
    """Tests for module exports."""

    def test_all_exports_from_module(self):
        """All expected items should be exported from the module."""
        from victor.framework.multi_agent.personas import (
            CommunicationStyle,
            ExpertiseLevel,
            PersonaTemplate,
            PersonaTraits,
        )

        assert CommunicationStyle is not None
        assert ExpertiseLevel is not None
        assert PersonaTemplate is not None
        assert PersonaTraits is not None

    def test_exports_from_package(self):
        """All expected items should be exported from the package."""
        from victor.framework.multi_agent import (
            CommunicationStyle,
            ExpertiseLevel,
            PersonaTemplate,
            PersonaTraits,
        )

        assert CommunicationStyle is not None
        assert ExpertiseLevel is not None
        assert PersonaTemplate is not None
        assert PersonaTraits is not None
