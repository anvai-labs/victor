"""StreamingChatExecutor delivers run_streaming's post-loop recovery request."""

import asyncio
from types import SimpleNamespace

from victor.agent.services import chat_stream_executor as cse
from victor.framework.agentic_loop import POST_LOOP_RECOVERY_KEY
from victor.providers.base import CompletionResponse, StreamChunk


class _Lifecycle:
    def __init__(self):
        self.cancelled = False

    def is_cancelled(self):
        return self.cancelled


class _Delivery:
    def sanitize(self, text, optional=False):
        return text.replace("<think>x</think>", "").strip()

    def content_chunk(self, content, *, is_final=False):
        return StreamChunk(content=content, is_final=is_final)


class _Executor:
    """The consumer's services, with the module helper bound like the run_unified call."""

    def __init__(self):
        self.services = SimpleNamespace(stream_lifecycle=_Lifecycle(), delivery=_Delivery())

    async def _deliver_post_loop_recovery(self, turn_executor, chunk, _stream_ctx=None):
        return await cse._deliver_post_loop_recovery(self.services, turn_executor, chunk)


def _executor():
    return _Executor()


class _TurnExecutor:
    def __init__(self, content="<think>x</think>Recovered answer.", delay=0.0, on_start=None):
        self.content, self.delay, self.on_start = content, delay, on_start
        self.reasons, self.finished = [], False

    async def synthesize_from_tool_evidence(self, *, recovery_reason=None):
        self.reasons.append(recovery_reason)
        if self.on_start:
            self.on_start()
        await asyncio.sleep(self.delay)
        self.finished = True
        return SimpleNamespace(response=CompletionResponse(content=self.content))


def _terminal(request):
    metadata = {"agentic_loop_success": False, "agentic_loop_error": "gave up"}
    if request is not None:
        metadata[POST_LOOP_RECOVERY_KEY] = request
    return StreamChunk(is_final=True, metadata=metadata)


async def test_request_is_synthesized_sanitized_and_marked_on_the_verdict():
    executor, turn_executor = _executor(), _TurnExecutor()
    terminal = _terminal({"kind": "synthesize", "reason": "gave up"})

    chunk = await executor._deliver_post_loop_recovery(turn_executor, terminal, SimpleNamespace())

    assert turn_executor.reasons == ["gave up"]
    assert chunk.content == "Recovered answer."  # through the delivery sanitizer
    assert POST_LOOP_RECOVERY_KEY not in terminal.metadata  # consumed, never forwarded
    assert terminal.metadata["agentic_loop_synthesis"] is True
    assert terminal.metadata["agentic_loop_synthesis_reason"] == "gave up"
    assert terminal.metadata["agentic_loop_success"] is False


async def test_no_request_means_no_synthesis():
    executor, turn_executor = _executor(), _TurnExecutor()
    terminal = _terminal(None)

    assert await executor._deliver_post_loop_recovery(turn_executor, terminal, None) is None
    assert turn_executor.reasons == []
    assert "agentic_loop_synthesis" not in terminal.metadata


async def test_cancelled_before_the_request_skips_synthesis():
    executor, turn_executor = _executor(), _TurnExecutor()
    executor.services.stream_lifecycle.cancelled = True

    chunk = await executor._deliver_post_loop_recovery(
        turn_executor, _terminal({"kind": "synthesize", "reason": "r"}), None
    )

    assert chunk is None and turn_executor.reasons == []


async def test_cancel_during_synthesis_abandons_the_call_promptly():
    executor = _executor()
    lifecycle = executor.services.stream_lifecycle
    turn_executor = _TurnExecutor(delay=30, on_start=lambda: setattr(lifecycle, "cancelled", True))

    chunk = await asyncio.wait_for(
        executor._deliver_post_loop_recovery(
            turn_executor, _terminal({"kind": "synthesize", "reason": "r"}), None
        ),
        timeout=2,
    )

    assert chunk is None
    assert turn_executor.finished is False  # the provider call was cancelled, not awaited


async def test_text_that_sanitizes_away_emits_nothing():
    executor, turn_executor = _executor(), _TurnExecutor(content="<think>x</think>")

    chunk = await executor._deliver_post_loop_recovery(
        turn_executor, _terminal({"kind": "synthesize", "reason": "r"}), None
    )

    assert chunk is None


class _FailingTurnExecutor:
    def __init__(self, exc):
        self.exc = exc

    async def synthesize_from_tool_evidence(self, *, recovery_reason=None):
        raise self.exc


async def test_failed_recovery_keeps_the_verdict_and_emits_nothing():
    executor = _executor()
    terminal = _terminal({"kind": "synthesize", "reason": "gave up"})

    chunk = await executor._deliver_post_loop_recovery(
        _FailingTurnExecutor(RuntimeError("429 rate limited")), terminal, None
    )

    assert chunk is None
    assert terminal.metadata["agentic_loop_error"] == "gave up"
    assert "agentic_loop_synthesis" not in terminal.metadata


async def test_policy_denial_during_recovery_still_surfaces():
    import pytest

    from victor.core.errors import ProviderPolicyError

    executor = _executor()
    with pytest.raises(ProviderPolicyError):
        await executor._deliver_post_loop_recovery(
            _FailingTurnExecutor(ProviderPolicyError("denied")),
            _terminal({"kind": "synthesize", "reason": "r"}),
            None,
        )


async def test_outer_cancellation_cancels_and_reaps_the_synthesis():
    executor, turn_executor = _executor(), _TurnExecutor(delay=30)
    outer = asyncio.ensure_future(
        executor._deliver_post_loop_recovery(
            turn_executor, _terminal({"kind": "synthesize", "reason": "r"}), None
        )
    )
    await asyncio.sleep(0.1)
    outer.cancel()

    import pytest

    with pytest.raises(asyncio.CancelledError):
        await outer
    assert turn_executor.finished is False
