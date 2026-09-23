from unittest.mock import AsyncMock

import pytest

from victor.coordination.formations.base import TeamContext
from victor.coordination.formations.group_chat import GroupChatFormation
from victor.teams.types import AgentMessage, MessageType, TeamFormation


@pytest.mark.parametrize(
    "value",
    [
        "prose",
        {},
        {"content": "x", "done": "yes", "handoff_to": None},
        {"content": "", "done": True, "handoff_to": None},
        {"content": "x", "done": False, "handoff_to": "unknown"},
        {"content": "x", "done": False, "handoff_to": "b"},
    ],
)
async def test_malformed_speech_fails_member(conversation_run, say, value):
    result, _, _ = await conversation_run(TeamFormation.GROUP_CHAT, {"a": [value], "b": [say()]})
    assert not result["success"]
    assert not result["member_results"]["a"].success


@pytest.mark.parametrize("target,done", [(None, False), ("a", False), ("b", True)])
async def test_invalid_handoff_contract(conversation_run, say, target, done):
    result, _, _ = await conversation_run(
        TeamFormation.HANDOFF, {"a": [say(target=target, done=done)], "b": [say()]}
    )
    assert not result["success"]


@pytest.mark.parametrize(
    "extra",
    [
        {"conversation_max_turns": 0},
        {"transcript_max_chars": 0},
        {"selector_func": "bad"},
        {"candidate_func": lambda *_: []},
        {"candidate_func": lambda *_: ["a", "a"]},
        {"candidate_func": lambda *_: ["unknown"]},
        {"selector_func": lambda *_: "unknown"},
        {"conversation_router_id": "unknown"},
        {"conversation_router_id": "a", "selector_func": lambda *_: "b"},
        {"termination_func": lambda *_: "yes"},
        {"transcript_max_chars": 1},
    ],
)
async def test_invalid_options_and_selection_fail_explicitly(conversation_run, say, extra):
    result, _, _ = await conversation_run(
        TeamFormation.GROUP_CHAT, {"a": [say()], "b": [say()]}, **extra
    )
    assert not result["success"]


async def test_callback_exception_is_no_fallback(conversation_run, say, caplog):
    def failed(*args):
        raise OSError("selector offline")

    result, calls, _ = await conversation_run(
        TeamFormation.GROUP_CHAT, {"a": [say()], "b": [say()]}, selector_func=failed
    )
    assert not result["success"] and not calls
    assert "selector offline" in caplog.text


async def test_checkpoint_is_rejected_before_execution():
    context = TeamContext("team", "group_chat")
    context.checkpoint_hook = AsyncMock()
    with pytest.raises(ValueError, match="durable"):
        await GroupChatFormation().execute(
            [], context, AgentMessage(sender_id="c", message_type=MessageType.TASK, content="x")
        )


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize(
    "mode,outcome",
    [
        ("group_chat", "valid"),
        ("group_chat_router", "valid"),
        ("debate", "valid"),
        ("handoff", "valid"),
        ("group_chat", "missing_usage"),
        ("group_chat", "changed_session"),
        ("group_chat", "malformed_last_turn"),
    ],
)
async def test_repeated_speakers_retain_opted_in_attempt_usage(monkeypatch, capture, mode, outcome):
    import json
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from victor.framework.teams import AgentTeam, TeamMemberSpec

    calls = {}
    # A provider may reuse nested metadata between calls; capture must snapshot it.
    counters = {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
    identifiers = {}

    async def spawn(**kwargs):
        name = kwargs["display_name"]
        count = calls.get(name, 0) + 1
        calls[name] = count
        counters.update(input_tokens=3 * count, output_tokens=2, total_tokens=3 * count + 2)
        details = {"session_id": "session-" + name, "usage": counters}
        if name == "first" and count == 1:
            if outcome == "missing_usage":
                details.pop("usage")
            if outcome == "changed_session":
                details["session_id"] = "different-session"
        if name == "controller":
            output = (
                {"selected_member_id": identifiers["first"], "verdict": "selected"}
                if mode == "debate"
                else {"speaker_id": identifiers["first" if count % 2 else "second"]}
            )
        else:
            done = name == "first" and count == 2
            output = {
                "content": f"{name}-{count}",
                "done": done,
                "handoff_to": (
                    identifiers["second" if name == "first" else "first"]
                    if mode == "handoff" and not done
                    else None
                ),
            }
        malformed = outcome == "malformed_last_turn" and name == "first" and count == 2
        return SimpleNamespace(
            success=True,
            summary="invalid JSON" if malformed else json.dumps(output),
            details=details,
            tool_calls_used=1,
            duration_seconds=0.5,
        )

    monkeypatch.setattr(
        "victor.agent.subagents.orchestrator.SubAgentOrchestrator",
        lambda *args: SimpleNamespace(spawn=AsyncMock(side_effect=spawn)),
    )
    names = ["first", "second"]
    if mode in {"debate", "group_chat_router"}:
        names.append("controller")
    team = await AgentTeam.create(
        MagicMock(),
        "conversation",
        "contribute",
        [TeamMemberSpec(role="executor", name=name, goal="contribute") for name in names],
        formation=TeamFormation.GROUP_CHAT if mode == "group_chat_router" else TeamFormation(mode),
        shared_context={"capture_member_usage": capture, "conversation_max_turns": 3},
    )
    identifiers.update({member.name: member.id for member in team._config.members})
    if mode in {"debate", "group_chat_router"}:
        role = "judge" if mode == "debate" else "router"
        team._config.shared_context[f"conversation_{role}_id"] = identifiers["controller"]
    result = await team.run()
    assert result.success == (
        outcome != "malformed_last_turn" and (not capture or outcome == "valid")
    )
    assert calls["first"] == 2
    for name, count in calls.items():
        member = result.member_results[identifiers[name]]
        assert member.tool_calls_used == count
        assert member.duration_seconds == count * 0.5
        assert member.metadata["conversation_turns"] == count
        bad = capture and name == "first" and outcome in {"missing_usage", "changed_session"}
        if bad:
            assert member.metadata["conversation_capture_error"] is True
            assert "usage" not in member.metadata
        elif capture:
            total_input = 3 * count * (count + 1) // 2
            assert member.metadata["usage"] == {
                "input_tokens": total_input,
                "output_tokens": 2 * count,
                "total_tokens": total_input + 2 * count,
            }
        if capture:
            attempts = member.metadata["conversation_attempts"]
            assert len(attempts) == count
            if not bad:
                assert attempts[0]["metadata"]["usage"]["input_tokens"] == 3
            if outcome == "malformed_last_turn" and name == "first":
                assert attempts[-1]["success"] is False
        else:
            assert "conversation_attempts" not in member.metadata
    if capture:
        first = result.member_results[identifiers["first"]]
        if not first.metadata.get("conversation_capture_error"):
            first.metadata["usage"]["input_tokens"] = 999
            assert (
                first.metadata["conversation_attempts"][-1]["metadata"]["usage"]["input_tokens"]
                == 6
            )
