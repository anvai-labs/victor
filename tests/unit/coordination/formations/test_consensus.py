"""Table-driven outcome contracts for consensus, reflection and parallel teams."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from victor.coordination.formations.consensus import ConsensusFormation
from victor.teams.types import MemberResult, TeamFormation
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
    "outputs, threshold, expected",
    [
        (["a", "b", "c"], 0.7, False),
        (["a", "b", "b"], 0.6, True),
        (["a", "a", "b"], 0.7, False),
        (["a", "a", "a"], 1.0, True),
        (["A", "a"], 1.0, False),
        ([], 0.7, False),
    ],
)
def test_consensus_compares_agreement_not_execution_success(outputs, threshold, expected):
    results = [
        MemberResult(member_id=str(i), success=True, output=value)
        for i, value in enumerate(outputs)
    ]
    assert ConsensusFormation(agreement_threshold=threshold)._check_consensus(results) is expected


def test_consensus_constructor_rejects_invalid_output_preview():
    """The preview bound is validated at construction (OOM follow-up)."""
    with pytest.raises(ValueError, match="positive output preview"):
        ConsensusFormation(output_preview_chars=0)
    with pytest.raises(ValueError, match="positive output preview"):
        ConsensusFormation(output_preview_chars="unbounded")  # type: ignore[arg-type]


async def test_consensus_rejects_invalid_preview_override_in_context():
    """A context-supplied preview override is re-validated in execute().

    The coordinator's _execute_with boundary converts the formation's
    ValueError into a failed result, so the rejection surfaces as
    success=False with the validation message — not a raised exception.
    """
    agents = [member("a", ["x"]), member("b", ["y"])]
    result = await coordinator(TeamFormation.CONSENSUS, agents).execute_task(
        "task",
        {"shared_state": {"consensus_output_preview_chars": 0}},
    )
    assert result["success"] is False
    assert "positive output preview" in str(result.get("error", ""))


def test_consensus_explicit_key_and_failed_member_denominator():
    results = [
        MemberResult(
            member_id="a", success=True, output="one", metadata={"consensus_key": "decision"}
        ),
        MemberResult(
            member_id="b", success=True, output="two", metadata={"consensus_key": "decision"}
        ),
    ]
    assert ConsensusFormation()._check_consensus(results)
    results.append(MemberResult(member_id="c", success=False, output=""))
    assert not ConsensusFormation()._check_consensus(results)


async def test_consensus_default_rounds_can_converge_after_disagreement():
    a = member("a", ["first", "agreed"])
    b = member("b", ["different", "agreed"])
    result = await coordinator(TeamFormation.CONSENSUS, [a, b]).execute_task("task", {})
    assert result["success"] is True
    assert result["consensus_rounds"] == 2
    assert result["consensus_achieved"] is True
    assert ConsensusFormation().max_rounds == 3


async def test_consensus_embeds_only_bounded_output_preview():
    """Echoing members must not compound the round task.

    A member that quotes the task it received makes the next round's task
    grow ~4x per round when the full output is embedded (14 GB by round ~11
    at max_rounds=15 — the 0.25.0 promotion OOM). Only a bounded preview of
    each output may be embedded.
    """
    seen_tasks: list = []
    agents = []
    for i in range(2):

        async def echo(task, context, i=i):
            seen_tasks.append(task)
            return f"echo-{i}: {task}"  # echoes the task; distinct per member

        agents.append(
            SimpleNamespace(
                id=f"a{i}",
                role="executor",
                receive_message=AsyncMock(),
                execute_task=AsyncMock(side_effect=echo),
            )
        )
    result = await coordinator(TeamFormation.CONSENSUS, agents).execute_task(
        "task",
        {"shared_state": {"consensus_max_rounds": 6, "consensus_output_preview_chars": 64}},
    )

    assert result["success"] is False  # echoes never converge
    assert len(seen_tasks) == 12  # 2 agents x 6 rounds: no early exit, no OOM
    # Each embedded output is clamped to the preview size, so a round task is
    # original + previews + JSON keys — bounded, not 4**round.
    worst = max(len(task) for task in seen_tasks)
    assert worst < 1_000, f"round task grew to {worst} chars"


@pytest.mark.parametrize(
    "tie_breaker, expected_success, expected_output",
    [
        (None, False, "a\n\nb"),
        ("supervisor", True, "a"),
    ],
)
async def test_consensus_exhaustion_and_supervisor_tie_break(
    tie_breaker, expected_success, expected_output
):
    members = [member("supervisor", ["a"]), member("b", ["b"])]
    shared = {"consensus_max_rounds": 1}
    if tie_breaker:
        shared["consensus_tie_breaker_id"] = tie_breaker
    result = await coordinator(TeamFormation.CONSENSUS, members).execute_task(
        "task", {"shared_state": shared}
    )
    assert result["success"] is expected_success
    assert result["consensus_achieved"] is False
    assert result["final_output"] == expected_output


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize(
    "outcome", ["converged", "exhausted", "missing_usage", "changed_session", "earlier_failure"]
)
async def test_consensus_retains_opted_in_whole_session_usage(monkeypatch, capture, outcome):
    from unittest.mock import MagicMock
    from victor.framework.teams import AgentTeam, TeamMemberSpec

    calls = {}

    async def spawn(**kwargs):
        identifier = kwargs["member_id"]
        count = calls.get(identifier, 0) + 1
        calls[identifier] = count
        output = "agreed" if count == 2 and outcome != "exhausted" else kwargs["display_name"]
        details = {
            "session_id": "session-" + identifier,
            "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5, "cached_tokens": 1},
        }
        if kwargs["display_name"] == "first" and count == 1:
            if outcome == "missing_usage":
                details.pop("usage")
            if outcome == "changed_session":
                details["session_id"] = "different-session"
        failed = outcome == "earlier_failure" and count == 1
        return SimpleNamespace(
            success=not failed,
            summary=output,
            details=details,
            error="failed attempt" if failed else None,
            tool_calls_used=1,
            duration_seconds=1.0,
        )

    monkeypatch.setattr(
        "victor.agent.subagents.orchestrator.SubAgentOrchestrator",
        lambda *args: SimpleNamespace(spawn=AsyncMock(side_effect=spawn)),
    )
    team = await AgentTeam.create(
        MagicMock(),
        "consensus",
        "agree",
        [TeamMemberSpec(role="executor", name=name, goal="agree") for name in ("first", "second")],
        formation=TeamFormation.CONSENSUS,
        shared_context={"capture_member_usage": capture, "consensus_max_rounds": 2},
    )
    result = await team.run()
    assert result.consensus_rounds == 2
    assert result.success == (outcome != "exhausted" and (not capture or outcome == "converged"))
    for member in result.member_results.values():
        bad = (
            capture
            and member.member_id == team._config.members[0].id
            and outcome in {"missing_usage", "changed_session"}
        )
        if bad:
            assert member.metadata["consensus_capture_error"] is True
            assert "usage" not in member.metadata
        else:
            multiplier = 2 if capture else 1
            assert member.metadata["usage"] == {
                "input_tokens": 3 * multiplier,
                "output_tokens": 2 * multiplier,
                "total_tokens": 5 * multiplier,
                "cached_tokens": multiplier,
            }
        assert member.tool_calls_used == (2 if capture else 1)
        assert member.duration_seconds == (2 if capture else 1)
        if capture:
            assert len(member.metadata["consensus_attempts"]) == 2
        else:
            assert "consensus_attempts" not in member.metadata


async def test_consensus_respects_member_concurrency_limit():
    import asyncio
    from victor.coordination.formations.base import TeamContext
    from victor.teams.types import AgentMessage, MessageType

    active = 0
    peak = 0

    class Agent:
        def __init__(self, member_id):
            self.id = member_id

        async def execute(self, task, context):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0)
            active -= 1
            return MemberResult(member_id=self.id, success=True, output="agreed")

    result = await ConsensusFormation(max_rounds=1).execute(
        [Agent(str(i)) for i in range(3)],
        TeamContext("t", "consensus", shared_state={"member_concurrency_limit": 1}),
        AgentMessage(sender_id="user", content="task", message_type=MessageType.TASK),
    )
    assert peak == 1
    assert len(result) == 3
    assert all(item.metadata["consensus_achieved"] for item in result)


async def test_consensus_self_cancelled_member_is_a_failed_result():
    import asyncio
    from victor.coordination.formations.base import TeamContext
    from victor.teams.types import AgentMessage, MessageType

    async def cancelled(task, context):
        raise asyncio.CancelledError("member stopped")

    result = await ConsensusFormation(max_rounds=1).execute(
        [SimpleNamespace(id="a", execute=cancelled)],
        TeamContext("t", "consensus"),
        AgentMessage(sender_id="user", content="task", message_type=MessageType.TASK),
    )
    assert len(result) == 1
    assert result[0].member_id == "a"
    assert result[0].success is False
    assert result[0].metadata["consensus_achieved"] is False


async def test_consensus_without_history_releases_completed_round_results():
    import weakref
    from victor.coordination.formations.base import TeamContext
    from victor.teams.types import AgentMessage, MessageType

    references = []
    live_at_round_start = []

    class Agent:
        def __init__(self, member_id):
            self.id = member_id

        async def execute(self, task, context):
            live_at_round_start.append(sum(ref() is not None for ref in references))
            result = MemberResult(
                member_id=self.id, success=True, output=f"{self.id}: {task.content}"
            )
            references.append(weakref.ref(result))
            return result

    results = await ConsensusFormation(max_rounds=15, output_preview_chars=64).execute(
        [Agent("a"), Agent("b")],
        TeamContext("t", "consensus"),
        AgentMessage(sender_id="user", content="task", message_type=MessageType.TASK),
    )
    assert len(results) == 2
    # One previous wave may remain referenced while its successor is admitted;
    # completed history must not grow with the number of rounds.
    assert max(live_at_round_start) <= 4


@pytest.mark.parametrize(
    "options",
    [
        {"max_rounds": True},
        {"max_rounds": 1.5},
        {"agreement_threshold": True},
        {"agreement_threshold": "0.7"},
        {"agreement_threshold": None},
        {"agreement_threshold": float("nan")},
        {"output_preview_chars": True},
    ],
)
def test_consensus_rejects_malformed_limit_configuration(options):
    with pytest.raises(ValueError, match="Consensus requires"):
        ConsensusFormation(**options)


@pytest.mark.parametrize(
    "options",
    [
        {"consensus_max_rounds": True},
        {"consensus_agreement_threshold": True},
        {"consensus_agreement_threshold": "0.7"},
        {"consensus_output_preview_chars": True},
    ],
)
async def test_consensus_rejects_malformed_context_limits_before_member_dispatch(options):
    agents = [member("a", ["yes"])]
    result = await coordinator(TeamFormation.CONSENSUS, agents).execute_task(
        "task", {"shared_state": options}
    )
    assert result["success"] is False
    assert "Consensus requires" in result.get("error", "")
    agents[0].execute_task.assert_not_called()


@pytest.mark.parametrize("rounds", [None, 15])
async def test_consensus_default_is_three_rounds_with_explicit_long_run_override(rounds):
    agents = [member("a", ["a"] * 20), member("b", ["b"] * 20)]
    context = {} if rounds is None else {"shared_state": {"consensus_max_rounds": rounds}}
    result = await coordinator(TeamFormation.CONSENSUS, agents).execute_task("task", context)
    expected = 3 if rounds is None else rounds
    assert result["success"] is False
    assert result["consensus_achieved"] is False
    assert result["consensus_rounds"] == expected
    assert all(agent.execute_task.await_count == expected for agent in agents)


def test_consensus_budget_defaults_agree_across_public_entry_points():
    import inspect
    from victor.framework.teams import AgentTeam
    from victor.framework.conversations.protocols import ConsensusProtocol

    preset_rounds = inspect.signature(AgentTeam.create_consensus_team).parameters["rounds"].default
    assert (
        ConsensusFormation().max_rounds
        == ConsensusProtocol().max_voting_rounds
        == preset_rounds
        == 3
    )
    assert ConsensusProtocol(max_voting_rounds=15).max_voting_rounds == 15
