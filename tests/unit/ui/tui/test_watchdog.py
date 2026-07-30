"""Unit tests for the stall watchdog (pure asyncio, no Textual)."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, List

import pytest

from victor.ui.tui.watchdog import with_stall_watchdog


async def _from_list(items: List[int]) -> AsyncIterator[int]:
    for item in items:
        yield item


async def test_passes_normal_events_through_unchanged() -> None:
    stalls: List[int] = []
    out: List[int] = []
    async for value in with_stall_watchdog(_from_list([1, 2, 3]), 1.0, lambda: stalls.append(1)):
        out.append(value)
    assert out == [1, 2, 3]
    assert stalls == []


async def test_fires_on_stall_then_resumes() -> None:
    stalls: List[int] = []
    resumes: List[int] = []

    async def slow() -> AsyncIterator[int]:
        yield 1
        await asyncio.sleep(0.25)
        yield 2

    out: List[int] = []
    async for value in with_stall_watchdog(
        slow(), 0.05, lambda: stalls.append(1), lambda: resumes.append(1)
    ):
        out.append(value)

    assert out == [1, 2]
    assert len(stalls) >= 1
    assert len(resumes) >= 1


async def test_propagates_source_error() -> None:
    async def boom() -> AsyncIterator[int]:
        yield 1
        raise RuntimeError("stream failed")

    seen: List[int] = []
    with pytest.raises(RuntimeError, match="stream failed"):
        async for value in with_stall_watchdog(boom(), 1.0, lambda: None):
            seen.append(value)
    assert seen == [1]


async def test_rejects_non_positive_timeout() -> None:
    with pytest.raises(ValueError):
        async for _ in with_stall_watchdog(_from_list([1]), 0.0, lambda: None):
            pass
