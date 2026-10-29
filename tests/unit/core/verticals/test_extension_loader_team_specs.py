# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Team specs built from an SDK VerticalDefinition's team metadata."""

from types import SimpleNamespace

from victor_contracts.core.types import TeamDefinition, TeamMemberDefinition, TeamMetadata
from victor_contracts.team_schema import TeamMemberSpec

from victor.core.verticals.extension_loader import _build_team_spec_provider_from_definition
from victor.teams.types import TeamAgentCategory


def _definition(teams, default_team=None):
    return SimpleNamespace(
        name="demo",
        team_metadata=TeamMetadata(teams=teams, default_team=default_team),
    )


def test_members_are_sdk_team_member_specs():
    team = TeamDefinition(
        team_id="review",
        name="Review",
        members=[
            TeamMemberDefinition(
                role="supervisor",
                goal="Coordinate",
                agent_category="supervisor",
                allowed_tools=["read_file"],
                memory_config={"enabled": True},
            ),
            TeamMemberDefinition(role="reviewer", goal="Review", expertise=["security"]),
        ],
    )
    provider = _build_team_spec_provider_from_definition(_definition([team], "review"))

    assert provider.get_default_team() == "review"
    lead, reviewer = provider.get_team_specs()["review"].members
    assert all(isinstance(m, TeamMemberSpec) for m in (lead, reviewer))
    assert lead.is_supervisor and lead.can_delegate and lead.max_delegation_depth >= 1
    assert lead.agent_category is TeamAgentCategory.SUPERVISOR
    assert lead.memory_config is not None and lead.memory_config.enabled
    assert reviewer.expertise == ["security"] and not reviewer.is_supervisor


def test_no_teams_means_no_provider():
    assert _build_team_spec_provider_from_definition(_definition([])) is None
