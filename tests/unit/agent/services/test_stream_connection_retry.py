"""Tests for turn-level retry on mid-stream provider disconnects.

Regression coverage for the dogfooding failure where a single ``httpx.ReadError``
mid-stream raised ``ProviderConnectionError`` and aborted the entire task,
discarding ~26 iterations of agent work. The streaming retry loop now treats a
mid-stream disconnect as a transient, bounded-retryable condition.
"""

import pytest

from victor.agent.services.chat_stream_helpers import ChatStreamHelperMixin
from victor.core.errors import ProviderConnectionError


class _RetryHarness(ChatStreamHelperMixin):
    """Minimal host exercising only the retry loop (no orchestrator needed)."""

    def __init__(self, fail_times: int):
        self._fail_times = fail_times
        self.calls = 0

    async def _stream_provider_response_inner(self, tools, provider_kwargs, stream_ctx):
        self.calls += 1
        if self.calls <= self._fail_times:
            raise ProviderConnectionError(message="connection dropped mid-stream", provider="zai")
        return ("recovered", None, 1.0, True)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    async def _instant(_seconds):
        return None

    monkeypatch.setattr("victor.agent.services.chat_stream_helpers.asyncio.sleep", _instant)


class TestStreamConnectionRetry:
    @pytest.mark.asyncio
    async def test_single_disconnect_is_retried(self):
        harness = _RetryHarness(fail_times=1)
        result = await harness._stream_with_rate_limit_retry(None, {}, None, max_retries=3)
        assert result == ("recovered", None, 1.0, True)
        assert harness.calls == 2  # one failure + one success

    @pytest.mark.asyncio
    async def test_multiple_transient_disconnects_recover(self):
        harness = _RetryHarness(fail_times=2)
        result = await harness._stream_with_rate_limit_retry(None, {}, None, max_retries=3)
        assert result[0] == "recovered"
        assert harness.calls == 3

    @pytest.mark.asyncio
    async def test_persistent_disconnect_eventually_raises(self):
        harness = _RetryHarness(fail_times=99)
        with pytest.raises(ProviderConnectionError):
            await harness._stream_with_rate_limit_retry(None, {}, None, max_retries=2)
        # max_retries + 1 attempts, then give up.
        assert harness.calls == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapped", [False, True])
async def test_gateway_disconnect_is_not_replayed(wrapped):
    from types import SimpleNamespace

    harness = _RetryHarness(fail_times=99)
    harness._orchestrator = SimpleNamespace(
        provider=SimpleNamespace(extra_config={"gateway": {"url": "https://gateway.test"}})
    )
    if wrapped:
        from victor.providers.factory import ManagedProvider

        harness._orchestrator.provider = ManagedProvider(harness._orchestrator.provider)
    with pytest.raises(ProviderConnectionError):
        await harness._stream_with_rate_limit_retry(None, {}, None)
    assert harness.calls == 1
