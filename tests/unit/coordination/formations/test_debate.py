from victor.teams.types import TeamFormation
from victor.coordination.formations.debate import DebateFormation


async def test_debate_dispatch_judges_once_after_contributions(conversation_run, say):
    result, calls, _ = await conversation_run(
        TeamFormation.DEBATE,
        {
            "a": [say("proposal")],
            "b": [say("counter")],
            "j": [{"selected_member_id": "b", "verdict": "supported"}],
        },
        conversation_judge_id="j",
        conversation_max_turns=4,
    )
    assert result["success"] and result["final_output"] == "supported"
    assert [identifier for identifier, _ in calls] == ["a", "b", "a", "b", "j"]
    assert len(calls[-1][1]["transcript"]) == 4
    assert DebateFormation().supports_durable_pause() is False


async def test_judge_cannot_select_nonparticipant(conversation_run, say):
    result, _, _ = await conversation_run(
        TeamFormation.DEBATE,
        {"a": [say()], "b": [say()], "j": [{"selected_member_id": "unknown", "verdict": "bad"}]},
        conversation_judge_id="j",
        conversation_max_turns=2,
    )
    assert not result["success"]
    assert not result["member_results"]["j"].success
