from victor.teams.types import TeamFormation
from victor.coordination.formations.handoff import HandoffFormation


async def test_peer_handoff_carries_transcript_and_uses_configured_start(conversation_run, say):
    result, calls, events = await conversation_run(
        TeamFormation.HANDOFF,
        {"a": [say("done", True)], "b": [say("transfer", target="a")]},
        handoff_start_id="b",
    )
    assert result["success"]
    assert [identifier for identifier, _ in calls] == ["b", "a"]
    assert calls[1][1]["transcript"][0]["handoff_to"] == "a"
    handoff = next(event for event in events if event.kind == "member_handoff")
    assert handoff.metadata == {"source_member_id": "b", "target_member_id": "a", "sequence": 0}
    assert HandoffFormation().supports_durable_pause() is False


async def test_cyclic_handoff_is_bounded(conversation_run, say):
    result, calls, _ = await conversation_run(
        TeamFormation.HANDOFF,
        {"a": [say(target="b")], "b": [say(target="a")]},
        conversation_max_turns=3,
    )
    assert not result["success"]
    assert len(calls) == 3
    assert result["shared_context"]["conversation_termination"] == "max_turns"
