"""Canonical role and supervisor-alias compatibility contracts (WS-I)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from victor.coordination.formations.base import TeamContext
from victor.coordination.formations.hierarchical import HierarchicalFormation
from victor.framework.teams import AgentTeam, TeamMemberSpec
from victor.teams.types import FormationRole, TeamFormation, normalize_supervisor_context
from victor.teams.unified_coordinator import UnifiedTeamCoordinator


@pytest.mark.parametrize(
    "canonical, legacy, expected",
    [
        (None, "old", None),
        ("new", "old", "new"),
        ("new", "new", "new"),
    ],
)
def test_canonical_supervisor_key_wins_and_alias_is_consumed(canonical, legacy, expected, caplog):
    state = {"explicit_supervisor_id": canonical, "explicit_manager_id": legacy}
    with caplog.at_level("WARNING"):
        normalize_supervisor_context(state)
    assert state == {"explicit_supervisor_id": expected}
    assert "deprecated" in caplog.text


def test_legacy_only_input_remains_supported_once(caplog):
    state = {"explicit_manager_id": "old"}
    with caplog.at_level("WARNING"):
        normalize_supervisor_context(state)
        normalize_supervisor_context(state)
    assert state == {"explicit_supervisor_id": "old"}
    assert len(caplog.records) == 1


async def test_coordinator_emits_one_key_and_does_not_mutate_caller_context():
    coord = UnifiedTeamCoordinator(lightweight_mode=True)
    item = SimpleNamespace(
        id="sup",
        role="executor",
        execute_task=AsyncMock(return_value="ok"),
        receive_message=AsyncMock(),
    )
    coord.set_supervisor(item)
    coord.set_formation(TeamFormation.SEQUENTIAL)
    caller_state = {"explicit_manager_id": "old"}
    result = await coord.execute_task("task", {"shared_state": caller_state})
    assert result["success"]
    assert result["shared_context"]["explicit_supervisor_id"] == "sup"
    assert "explicit_manager_id" not in result["shared_context"]
    assert caller_state == {"explicit_manager_id": "old"}
    assert coord.manager is coord.supervisor is item
    coord.clear()
    assert coord.supervisor is None


def test_direct_hierarchy_accepts_legacy_alias():
    members = [
        SimpleNamespace(id="first", is_supervisor=False, can_delegate=False),
        SimpleNamespace(id="selected", is_supervisor=False, can_delegate=False),
    ]
    context = TeamContext("t", "hierarchical", {"explicit_manager_id": "selected"})
    supervisor, other = HierarchicalFormation()._resolve_supervisor(members, context)
    assert supervisor.id == "selected"
    assert [item.id for item in other] == ["first"]
    assert context.shared_state == {"explicit_supervisor_id": "selected"}


@pytest.mark.parametrize("role", list(FormationRole))
def test_canonical_formation_role_round_trip(role):
    spec = TeamMemberSpec(role="executor", goal="task", formation_role=role.value)
    assert spec.to_team_member().formation_role == role.value


@pytest.mark.parametrize("alias", ["manager", "planner", "evaluator", "worker", "lead"])
def test_formation_roles_reject_synonyms_and_domain_roles(alias):
    with pytest.raises(ValueError):
        TeamMemberSpec(role="executor", goal="task", formation_role=alias).to_team_member()


async def test_review_and_reflection_presets_bind_distinct_roles_without_mutating_specs():
    writer = TeamMemberSpec(role="executor", goal="write")
    evaluator = TeamMemberSpec(role="reviewer", goal="assess")
    review = await AgentTeam.create_review_team(
        SimpleNamespace(), "one pass", "task", writer=writer, reviewer=evaluator
    )
    reflection = await AgentTeam.create_reflection_team(
        SimpleNamespace(), "iterate", "task", generator=writer, critic=evaluator
    )
    assert review._config.members[1].formation_role == "reviewer"
    assert reflection._config.members[1].formation_role == "critic"
    assert writer.formation_role is evaluator.formation_role is None
