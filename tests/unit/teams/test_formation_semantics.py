"""Table-driven outcome contracts for consensus, reflection and parallel teams."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from victor.coordination.formations.base import TeamContext
from victor.coordination.formations.reflection import ReflectionFormation
from victor.framework.teams import AgentTeam, TeamMemberSpec
from victor.teams.types import AgentMessage, MessageType, TeamFormation
from victor.teams.unified_coordinator import UnifiedTeamCoordinator


def member(name, responses):
    return SimpleNamespace(
        id=name,
        role="executor",
        receive_message=AsyncMock(),
        execute_task=AsyncMock(side_effect=responses),
    )


def coordinator(formation, members):
    coord = UnifiedTeamCoordinator(lightweight_mode=True)
    coord.set_formation(formation)
    for item in members:
        coord.add_member(item)
    return coord


@pytest.mark.parametrize(
    "feedback, satisfied",
    [
        ('{"verdict":"satisfied","feedback":"Done"}', True),
        ('{"verdict":"needs_work","feedback":"Looks good, but missing tests"}', False),
    ],
)
def test_reflection_json_contract(feedback, satisfied):
    assert ReflectionFormation()._is_satisfied(feedback, "json") is satisfied


@pytest.mark.parametrize(
    "feedback",
    [
        "looks good",
        "VERDICT: SATISFIED",
        "{}",
        "[]",
        "null",
        '{"verdict":"satisfied","feedback":2}',
        '{"verdict":"satisfied","feedback":"ok","extra":true}',
        '{"verdict":"maybe","feedback":"ok"}',
    ],
)
async def test_invalid_structured_verdict_stops_without_extra_generation(feedback, caplog):
    generator = SimpleNamespace(id="generator", execute=AsyncMock(return_value="draft"))
    critic = SimpleNamespace(id="critic", execute=AsyncMock(return_value=feedback))
    ctx = TeamContext(
        "t",
        "reflection",
        {"generator": generator, "critic": critic, "reflection_verdict_format": "json"},
    )
    with caplog.at_level("WARNING"):
        results = await ReflectionFormation().execute(
            [], ctx, AgentMessage(sender_id="test", content="task", message_type=MessageType.TASK)
        )
    assert results[0].success is False
    assert results[0].metadata["verdict_contract_error"] is True
    generator.execute.assert_awaited_once()
    assert "Invalid reflection verdict" in caplog.text


async def test_parallel_failed_deliverable_is_visible_and_retry_is_opt_in():
    a = member("a", [{"success": True, "output": "delivered"}])
    b = member("b", [{"success": False, "error": "missing file"}])
    result = await coordinator(TeamFormation.PARALLEL, [a, b]).execute_task("task", {})
    assert result["success"] is True
    assert result["final_output"] == "delivered\n\nMember failures:\n- b: missing file"
    assert result["member_results"]["b"].success is False
    b.execute_task.assert_awaited_once()


async def test_parallel_retry_counts_attempt_costs_and_keeps_successful_member_single_pass():
    a = member("a", [{"success": True, "output": "A", "tool_calls_used": 2}])
    b = member(
        "b",
        [
            {"success": False, "error": "transient", "tool_calls_used": 1},
            {"success": True, "output": "B", "tool_calls_used": 3},
        ],
    )
    result = await coordinator(TeamFormation.PARALLEL, [a, b]).execute_task(
        "task", {"shared_state": {"parallel_member_retries": 1}}
    )
    assert result["final_output"] == "A\n\nB"
    assert result["total_tool_calls"] == 6
    assert result["member_results"]["b"].metadata["execution_attempts"] == 2
    a.execute_task.assert_awaited_once()
    assert b.execute_task.await_count == 2


async def test_parallel_awaiting_approval_is_never_retried():
    item = member("a", [{"success": False, "metadata": {"awaiting_approval": True}}])
    await coordinator(TeamFormation.PARALLEL, [item]).execute_task(
        "task", {"shared_state": {"parallel_member_retries": 3}}
    )
    item.execute_task.assert_awaited_once()


async def test_presets_expose_contracts_without_mutating_legacy_defaults():
    specs = [TeamMemberSpec(role="executor", goal="task")]
    team = await AgentTeam.create_consensus_team(
        SimpleNamespace(),
        "test",
        "task",
        specs,
        supervisor=TeamMemberSpec(role="executor", goal="decide"),
    )
    assert team._config.shared_context["consensus_max_rounds"] == 3
    assert team._config.shared_context["consensus_tie_breaker_id"] == team._config.members[0].id
    for verdict_format in ["legacy", "json"]:
        team = await AgentTeam.create_reflection_team(
            SimpleNamespace(),
            "test",
            "task",
            generator=specs[0],
            critic=TeamMemberSpec(role="reviewer", goal="critique"),
            verdict_format=verdict_format,
        )
        assert (
            team._config.shared_context.get("reflection_verdict_format", "legacy") == verdict_format
        )


async def test_parallel_retry_rolls_up_neutral_usage():
    item = member(
        "a",
        [
            {
                "success": False,
                "metadata": {"usage": {"input_tokens": 11, "output_tokens": 2, "total_tokens": 13}},
            },
            {
                "success": True,
                "output": "done",
                "metadata": {"usage": {"input_tokens": 17, "output_tokens": 3, "total_tokens": 20}},
            },
        ],
    )
    result = await coordinator(TeamFormation.PARALLEL, [item]).execute_task(
        "work", {"parallel_member_retries": 1}
    )
    assert result["member_results"]["a"].metadata["usage"] == {
        "input_tokens": 28,
        "output_tokens": 5,
        "total_tokens": 33,
    }
