"""Empty-response recovery must keep tool-call history consistent.

Regression for session modality-doc-review-fixes-b4e87728 (2026-07-24): the
recovery path returned recovered tool calls WITHOUT recording the assistant
tool_calls message, so the executed tools' ``role=tool`` results became
orphaned in history. The pairing repairs then stranded one side or the other
on every later turn until Moonshot rejected the payload with a non-retryable
400 and the turn died with an empty response.
"""

from types import SimpleNamespace

import pytest

from victor.agent.services.chat_stream_helpers import ChatStreamHelperMixin


class _Helper(ChatStreamHelperMixin):
    def __init__(self, orchestrator):
        self._orchestrator = orchestrator


def _recovery_orch(chunks, added):
    async def _stream(**kwargs):
        for chunk in chunks:
            yield chunk

    return SimpleNamespace(
        add_message=lambda role, content, **kwargs: added.append((role, content, kwargs)),
        thinking=False,
        model="kimi-k3",
        max_tokens=4096,
        get_assembled_messages=lambda **kwargs: [{"role": "user", "content": "q"}],
        provider=SimpleNamespace(stream=_stream),
        sanitizer=SimpleNamespace(sanitize=lambda c: c),
        _chunk_generator=SimpleNamespace(
            generate_content_chunk=lambda text, is_final=False: SimpleNamespace(
                content=text, is_final=is_final
            )
        ),
    )


def _stream_ctx():
    return SimpleNamespace(
        provider_kwargs={},
        user_message="q",
        goals=None,
        planned_tools=None,
    )


@pytest.mark.asyncio
async def test_recovered_tool_calls_record_assistant_message():
    """The recovered tool calls must land in history as an assistant message."""
    tool_calls = [{"id": "call_1", "function": {"name": "shell", "arguments": "{}"}}]
    added = []
    orch = _recovery_orch([SimpleNamespace(content="", tool_calls=tool_calls)], added)
    helper = _Helper(orch)

    success, recovered, final_chunk = await helper._handle_empty_response_recovery(
        _stream_ctx(), tools=[]
    )

    assert success is True
    assert recovered == tool_calls
    assert final_chunk is None
    assistant_entries = [entry for entry in added if entry[0] == "assistant"]
    assert len(assistant_entries) == 1
    role, content, kwargs = assistant_entries[0]
    assert content == ""
    assert kwargs.get("tool_calls") == tool_calls


@pytest.mark.asyncio
async def test_recovered_content_still_records_assistant_message():
    """The pre-existing content path keeps recording the assistant message."""
    added = []
    orch = _recovery_orch([SimpleNamespace(content="recovered answer", tool_calls=None)], added)
    helper = _Helper(orch)

    success, recovered, final_chunk = await helper._handle_empty_response_recovery(
        _stream_ctx(), tools=[]
    )

    assert success is True
    assert recovered is None
    assert final_chunk is not None and final_chunk.content == "recovered answer"
    assistant_entries = [entry for entry in added if entry[0] == "assistant"]
    assert len(assistant_entries) == 1
    assert assistant_entries[0][1] == "recovered answer"


def test_payload_shape_logged_on_client_error(caplog):
    """A 4xx provider rejection logs the per-message payload shape."""
    error = SimpleNamespace(status_code=400)
    messages = [
        {"role": "system", "content": "sys"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_9", "function": {"name": "read"}}],
        },
        {"role": "tool", "content": "result", "tool_call_id": "call_9"},
        {"role": "tool", "content": "orphan", "tool_call_id": "call_lost"},
    ]

    with caplog.at_level("WARNING"):
        ChatStreamHelperMixin._log_client_error_payload_shape(error, messages)

    record = next(r for r in caplog.records if "[provider-4xx]" in r.getMessage())
    text = record.getMessage()
    assert "status=400" in text
    assert "tool_calls=['call_9']" in text
    assert "tool_call_id=call_lost" in text


@pytest.mark.parametrize("status", [None, 200, 429, 500])
def test_payload_shape_not_logged_for_non_client_errors(caplog, status):
    """Transient statuses (5xx, 429) and unknown errors stay quiet."""
    error = SimpleNamespace(status_code=status)

    with caplog.at_level("WARNING"):
        ChatStreamHelperMixin._log_client_error_payload_shape(
            error, [{"role": "user", "content": "q"}]
        )

    assert not any("[provider-4xx]" in r.getMessage() for r in caplog.records)


def _capture_orch(chunks, added, stream_kwargs):
    async def _stream(**kwargs):
        stream_kwargs.append(kwargs)
        for chunk in chunks:
            yield chunk

    orch = _recovery_orch(chunks, added)
    orch.provider = SimpleNamespace(stream=_stream)
    return orch


@pytest.mark.asyncio
async def test_reasoning_exhaustion_raises_recovery_max_tokens():
    """A reasoning-only empty stream must retry with a larger completion budget.

    Regression: kimi-k3 burned the whole max_tokens budget on reasoning deltas
    (~112s, zero content) and identical-parameter recovery failed identically.
    """
    added = []
    stream_kwargs = []
    orch = _capture_orch(
        [SimpleNamespace(content="recovered", tool_calls=None)], added, stream_kwargs
    )
    ctx = _stream_ctx()
    ctx.empty_stream_diagnostics = {"reasoning_chars": 8321, "stop_reason": "length"}
    helper = _Helper(orch)

    success, _, final_chunk = await helper._handle_empty_response_recovery(ctx, tools=[])

    assert success is True
    assert final_chunk is not None
    assert stream_kwargs[0]["max_tokens"] == 4096 * 4


@pytest.mark.asyncio
async def test_reasoning_exhaustion_max_tokens_capped():
    """The recovery budget bump is capped at 32768."""
    added = []
    stream_kwargs = []
    orch = _capture_orch(
        [SimpleNamespace(content="recovered", tool_calls=None)], added, stream_kwargs
    )
    orch.max_tokens = 16384
    ctx = _stream_ctx()
    ctx.empty_stream_diagnostics = {"reasoning_chars": 100, "stop_reason": "length"}
    helper = _Helper(orch)

    await helper._handle_empty_response_recovery(ctx, tools=[])

    assert stream_kwargs[0]["max_tokens"] == 32768


@pytest.mark.asyncio
async def test_no_diagnostics_keeps_base_max_tokens():
    """Without reasoning evidence the recovery keeps the configured budget."""
    added = []
    stream_kwargs = []
    orch = _capture_orch(
        [SimpleNamespace(content="recovered", tool_calls=None)], added, stream_kwargs
    )
    helper = _Helper(orch)

    await helper._handle_empty_response_recovery(_stream_ctx(), tools=[])

    assert stream_kwargs[0]["max_tokens"] == 4096
