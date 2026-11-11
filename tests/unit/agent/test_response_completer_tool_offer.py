"""Text-forcing completer calls keep the stable tool offer on tool history."""

from unittest.mock import AsyncMock, MagicMock

from victor.agent.response_completer import ResponseCompleter, ToolFailureContext
from victor.agent.services.tool_selection_runtime import TEXT_ONLY_INSTRUCTION
from victor.providers.base import CompletionResponse, Message

OFFER = [MagicMock(name="read"), MagicMock(name="grep")]


def _provider(*responses):
    provider = MagicMock()
    provider.chat = AsyncMock(side_effect=list(responses))
    return provider


def _text(content="A complete final answer."):
    return CompletionResponse(content=content, role="assistant")


SYSTEM = Message(role="system", content="You are Victor.")


async def _ensure(provider, **kwargs):
    return await ResponseCompleter(provider).ensure_response(
        messages=[SYSTEM, Message(role="user", content="q")],
        model="m",
        temperature=0.1,
        max_tokens=100,
        **kwargs,
    )


def _system_texts(call):
    return [m.content for m in call.kwargs["messages"] if m.role == "system"]


def _assert_text_only_tail(call):
    sent = call.kwargs["messages"]
    # The instruction is a trailing user turn; the session system prompt is the only
    # system message (single-system-block adapters keep the last one they see).
    assert sent[-1].role == "user" and TEXT_ONLY_INSTRUCTION in sent[-1].content
    assert _system_texts(call) == [SYSTEM.content]


async def test_retry_forwards_offer_and_text_only_instruction_on_every_attempt():
    provider = _provider(_text(""), _text())

    result = await _ensure(provider, tools=OFFER)

    assert result.content == "A complete final answer."
    for call in provider.chat.await_args_list:
        assert call.kwargs["tools"] is OFFER
        _assert_text_only_tail(call)


async def test_retry_without_offer_keeps_tool_free_request_unchanged():
    provider = _provider(_text())

    await _ensure(provider)

    call = provider.chat.await_args
    assert call.kwargs["tools"] is None
    # Attempt 0 without an offer adds no instruction (pre-existing request shape).
    assert call.kwargs["messages"] == [SYSTEM, Message(role="user", content="q")]


async def test_retry_without_offer_keeps_system_role_recovery_prompts():
    provider = _provider(_text(""), _text())

    await _ensure(provider)

    retry = provider.chat.await_args_list[1]
    assert retry.kwargs["tools"] is None
    assert retry.kwargs["messages"][-1].role == "system"
    assert TEXT_ONLY_INSTRUCTION not in retry.kwargs["messages"][-1].content


async def test_tool_failure_path_forwards_offer_and_instruction():
    provider = _provider(_text())
    failure = ToolFailureContext(failed_tools=[{"name": "read", "error": "nope"}])

    await _ensure(provider, tools=OFFER, failure_context=failure)

    call = provider.chat.await_args
    assert call.kwargs["tools"] is OFFER
    _assert_text_only_tail(call)
    assert "nope" in call.kwargs["messages"][-1].content  # failure summary kept


async def test_tool_call_reply_is_not_accepted_as_the_answer():
    # With an offer on the wire the model may still call a tool; that reply carries
    # no text and must fall through to the next attempt, never execute or complete.
    tool_only = CompletionResponse(
        content="",
        role="assistant",
        tool_calls=[{"id": "c1", "name": "read", "arguments": {}}],
    )
    provider = _provider(tool_only, _text())

    result = await _ensure(provider, tools=OFFER)

    assert result.content == "A complete final answer."
    assert result.retries_used == 1
