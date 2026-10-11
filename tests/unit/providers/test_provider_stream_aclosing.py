# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests: intermediate provider stream wrappers must cascade close.

A stream wrapper that re-yields `async for chunk in inner: yield chunk` without
`contextlib.aclosing(inner)` does NOT finalize the inner generator when the wrapper itself
is aclosed on an early consumer break (aclose does not cascade GeneratorExit into a
sub-generator iterated via `async for`). That left the underlying httpx byte stream
suspended and finalized off-task by GC — the source of "async generator ignored
GeneratorExit" / "exit cancel scope in a different task" warnings every turn.
"""

import contextlib
import asyncio
from unittest.mock import MagicMock

import pytest

from victor.providers.base import Message, StreamChunk
from victor.providers.cerebras_provider import CerebrasProvider
from victor.providers.deepseek_provider import DeepSeekProvider
from victor.providers.sandhi_transport import SandhiTypedProviderMixin
from victor.providers.resilience import ProviderRetryConfig, ResilientProvider


def _instrumented_inner(closed: dict):
    async def _stream(*_args, **_kwargs):
        try:
            for i in range(10):
                yield StreamChunk(content=f"chunk{i}")
        finally:
            closed["count"] = closed.get("count", 0) + 1

    return _stream


def _resilient_with(closed: dict) -> ResilientProvider:
    provider = MagicMock()
    provider.name = "fake"
    provider.stream = _instrumented_inner(closed)
    return ResilientProvider(provider, retry_config=ProviderRetryConfig(max_retries=0))


async def test_resilient_stream_cascades_close_on_early_break():
    """Early break must finalize the inner provider stream exactly once (the loop case)."""
    closed: dict = {}
    resilient = _resilient_with(closed)

    agen = resilient.stream(messages=[], model="m")
    async with contextlib.aclosing(agen):
        async for _chunk in agen:
            break  # tool-call-style early break — closes the resilient wrapper

    # Without the aclosing fix in ResilientProvider.stream this stays 0 (inner orphaned → GC).
    assert closed.get("count") == 1


async def test_resilient_stream_cascades_close_on_full_consume():
    closed: dict = {}
    resilient = _resilient_with(closed)

    agen = resilient.stream(messages=[], model="m")
    chunks = []
    async with contextlib.aclosing(agen):
        async for chunk in agen:
            chunks.append(chunk)

    assert len(chunks) == 10
    assert closed.get("count") == 1


@pytest.mark.parametrize("provider_cls", (DeepSeekProvider, CerebrasProvider))
@pytest.mark.parametrize("termination", ("early_close", "cancel", "cleanup_error"))
async def test_structured_provider_wrappers_close_typed_python_stream(
    provider_cls, termination, monkeypatch
):
    """Exercise both the concrete wrapper and shared HTTPx-policy wrapper."""
    closed = 0
    waiting = asyncio.Event()

    async def upstream(request):
        nonlocal closed
        try:
            yield StreamChunk(content="first")
            waiting.set()
            await asyncio.Event().wait()
        finally:
            closed += 1
            if termination == "cleanup_error":
                raise RuntimeError("cleanup failure")

    provider = provider_cls(api_key="test-key", text_response_mode="structured")
    monkeypatch.setattr(provider, "_sandhi_stream", upstream)
    stream = provider.stream([Message(role="user", content="hello")], model="qwen-3")
    assert (await asyncio.wait_for(anext(stream), timeout=1)).content == "first"
    if termination == "cancel":
        pending = asyncio.create_task(anext(stream))
        await asyncio.wait_for(waiting.wait(), timeout=1)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
    elif termination == "cleanup_error":
        with pytest.raises(RuntimeError, match="cleanup failure"):
            await stream.aclose()
    else:
        await stream.aclose()
    assert closed == 1
    await stream.aclose()
    assert closed == 1


async def test_typed_stream_generator_exit_is_not_mapped_to_provider_error(monkeypatch):
    """Python close is not a provider failure; this does not prove origin abort."""

    class Handle:
        def stream_json(self, *args):
            async def events():
                yield '{"event": "text_delta", "delta": "first"}'

            return events()

    provider = DeepSeekProvider(api_key="test-key")
    monkeypatch.setattr(provider, "_typed_provider", lambda model: Handle())
    stream = SandhiTypedProviderMixin._sandhi_stream(provider, {"model": "deepseek-chat"})
    assert (await anext(stream)).content == "first"
    await stream.aclose()
