"""Interop tests for SDK-owned team schema contracts."""

from victor.framework.team_schema import (
    RoleConfig as CoreRoleConfig,
    TeamSpec as CoreTeamSpec,
    create_team_spec as core_create_team_spec,
)
from victor_contracts.team_schema import (
    MemoryConfig,
    RoleConfig as SdkRoleConfig,
    TeamAgentCategory,
    TeamFormation,
    TeamMemberSpec,
    TeamSpec as SdkTeamSpec,
    create_team_spec as sdk_create_team_spec,
)


def test_team_schema_identity_is_shared() -> None:
    from victor.teams import types

    assert CoreRoleConfig is SdkRoleConfig
    assert CoreTeamSpec is SdkTeamSpec
    assert core_create_team_spec is sdk_create_team_spec
    assert types.MemoryConfig is MemoryConfig
    assert types.TeamAgentCategory is TeamAgentCategory


def test_core_team_schema_factory_accepts_sdk_member_specs() -> None:
    spec = core_create_team_spec(
        name="Research Team",
        description="Investigates issues",
        vertical="research",
        formation=TeamFormation.SEQUENTIAL,
        members=[TeamMemberSpec(role="researcher", goal="Investigate")],
    )

    assert isinstance(spec, SdkTeamSpec)
    assert spec.members[0].goal == "Investigate"
