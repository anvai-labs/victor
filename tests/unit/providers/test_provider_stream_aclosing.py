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
from unittest.mock import MagicMock

from victor.providers.base import StreamChunk
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
