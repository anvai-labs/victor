from victor.teams.types import TeamFormation
from victor.coordination.formations.group_chat import GroupChatFormation


async def test_round_robin_transcript_dispatch_metrics_and_termination(conversation_run, say):
    result, calls, events = await conversation_run(
        TeamFormation.GROUP_CHAT,
        {"a": [say("first"), say("final", True)], "b": [say("reply")]},
        conversation_max_turns=3,
    )
    assert result["success"]
    assert [identifier for identifier, _ in calls] == ["a", "b", "a"]
    assert calls[1][1]["transcript"][0]["content"] == "first"
    assert result["member_results"]["a"].tool_calls_used == 4
    assert result["member_results"]["a"].duration_seconds == 1
    assert result["final_output"] == "final"
    assert result["shared_context"]["conversation_termination"] == "done"
    assert len([event for event in events if event.kind == "member_spoke"]) == 3
    assert GroupChatFormation().supports_durable_pause() is False


async def test_router_and_programmatic_selection(conversation_run, say):
    result, calls, _ = await conversation_run(
        TeamFormation.GROUP_CHAT,
        {"a": [say()], "b": [say(done=True)], "r": [{"speaker_id": "b"}]},
        conversation_router_id="r",
    )
    assert result["success"]
    assert [identifier for identifier, _ in calls] == ["r", "b"]

    async def candidates(transcript, members):
        assert isinstance(transcript, tuple)
        return ["b"]

    result, calls, _ = await conversation_run(
        TeamFormation.GROUP_CHAT,
        {"a": [say()], "b": [say()]},
        candidate_func=candidates,
        selector_func=lambda transcript, members: members[0],
        termination_func=lambda transcript: len(transcript) == 1,
    )
    assert result["success"] and calls[0][0] == "b"
    assert result["shared_context"]["conversation_termination"] == "predicate"
