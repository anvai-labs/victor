# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Stall watchdog for the TUI event stream.

Wraps an async event source so that a model/agent that goes quiet surfaces as a
visible "waiting…" state instead of a silent frozen terminal (ADR-021; cf. the
wedged-loop failure mode in TD-20). The wrapper never drops or reorders items and
re-raises source errors unchanged.

Implementation note: we pump the source in a background task onto a queue and
time out on ``queue.get`` rather than on ``source.__anext__`` directly. Timing
out ``__anext__`` would *cancel* the in-flight pull and corrupt the underlying
async generator; timing out a queue read is side-effect-free and leaves the pump
untouched, so the source resumes cleanly once it produces again.

Pure asyncio; imports nothing from Textual and is unit-testable standalone.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import AsyncIterator, Callable, Optional, TypeVar

T = TypeVar("T")

StallCallback = Callable[[], None]

# Sentinels distinguishing normal completion from source failure on the queue.
_DONE = object()
_ERROR = object()


async def with_stall_watchdog(
    source: AsyncIterator[T],
    timeout: float,
    on_stall: StallCallback,
    on_resume: Optional[StallCallback] = None,
) -> AsyncIterator[T]:
    """Yield from ``source``, flagging stalls longer than ``timeout`` seconds.

    Args:
        source: The async event source to relay (e.g. mapped stream events).
        timeout: Seconds of silence before ``on_stall`` fires. Must be > 0.
        on_stall: Called once when the source first goes quiet past ``timeout``.
        on_resume: Called once when items flow again after a stall.

    Yields:
        Each item from ``source`` in order, unmodified.

    Raises:
        Any exception raised by ``source`` (re-raised after the pump drains).
    """
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    queue: asyncio.Queue[object] = asyncio.Queue()
    error: list[BaseException] = []

    async def _pump() -> None:
        try:
            async for item in source:
                await queue.put(item)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 - relayed to the consumer verbatim
            error.append(exc)
            await queue.put(_ERROR)
        else:
            await queue.put(_DONE)

    pump = asyncio.create_task(_pump())
    stalled = False
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout)
            except asyncio.TimeoutError:
                if not stalled:
                    stalled = True
                    on_stall()
                continue
            if stalled:
                stalled = False
                if on_resume is not None:
                    on_resume()
            if item is _DONE:
                return
            if item is _ERROR:
                raise error[0]
            yield item  # type: ignore[misc]
    finally:
        pump.cancel()
        with contextlib.suppress(BaseException):
            await pump
