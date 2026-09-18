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
