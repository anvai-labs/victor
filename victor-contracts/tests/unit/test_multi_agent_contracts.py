"""Tests for SDK-owned multi-agent contracts."""

import pytest

from victor_contracts.multi_agent import (
    CommunicationStyle,
    PersonaTemplate,
    PersonaTraits,
    TaskAssignmentStrategy,
    TeamAgentCategory,
    TeamFormation,
    TeamMember,
    TeamSpec,
    TeamTemplate,
    TeamTopology,
)
from victor_contracts.team_schema import TeamFormation as SchemaTeamFormation
from victor_contracts.team_schema import TeamMemberSpec as SchemaTeamMemberSpec
from victor_contracts.multi_agent import TeamMemberSpec


def test_persona_template_creates_overridden_persona() -> None:
    template = PersonaTemplate(
        base_traits=PersonaTraits(
            name="Base",
            role="reviewer",
            description="Base persona",
            communication_style=CommunicationStyle.TECHNICAL,
        ),
        overrides={"verbosity": 0.8},
    )

    created = template.create(name="Derived")

    assert created.name == "Derived"
    assert created.verbosity == 0.8


def test_team_template_validates_threshold() -> None:
    with pytest.raises(ValueError):
        TeamTemplate(
            name="Bad",
            description="Bad config",
            topology=TeamTopology.PIPELINE,
            assignment_strategy=TaskAssignmentStrategy.SKILL_MATCH,
            escalation_threshold=1.5,
        )


def test_team_spec_reports_missing_slots() -> None:
    spec = TeamSpec(
        template=TeamTemplate(
            name="Review",
            description="Review team",
            member_slots={"reviewer": 2},
        ),
        members=[],
    )

    assert spec.validate_slots() == ["Role 'reviewer' requires 2 members, got 0"]


def test_multi_agent_reuses_canonical_team_schema_types() -> None:
    assert TeamFormation is SchemaTeamFormation
    assert TeamMemberSpec is SchemaTeamMemberSpec


def test_persona_team_maps_topology_and_leader_to_canonical_semantics() -> None:
    persona = PersonaTraits(
        name="Lead",
        role="planner",
        description="Coordinates work",
    )
    member = TeamMember(
        persona=persona,
        role_in_team="planner",
        is_leader=True,
    )
    spec = TeamSpec(
        template=TeamTemplate(
            name="Supervised",
            description="Supervised team",
            topology=TeamTopology.HUB_SPOKE,
        ),
        members=[member],
    )

    assert spec.formation is TeamFormation.HIERARCHICAL
    assert spec.supervisor is member
    assert member.agent_category is TeamAgentCategory.SUPERVISOR


def test_runtime_persona_provider_hook_removed() -> None:
    """get_runtime_persona_provider was deprecated in 0.13.0 and removed in 0.14.0."""
    import victor_contracts
    from victor_contracts import multi_agent

    assert not hasattr(multi_agent, "get_runtime_persona_provider")
    assert "get_runtime_persona_provider" not in multi_agent.__all__
    assert not hasattr(victor_contracts, "get_runtime_persona_provider")
