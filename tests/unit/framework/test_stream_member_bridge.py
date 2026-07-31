"""ADR-023 increment 4: the teams->client-stream bridge in ``stream_with_events``.

These tests exercise the merge seam with a fake orchestrator (no real LLM). A team runs
"inside" the stream by reading ``current_member_sink`` from within the fake ``stream_chat``
and emitting member lifecycle events, which must interleave into the yielded event stream.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, AsyncIterator, List, Optional

from victor.framework._internal import stream_with_events
from victor.framework.events import EventType
from victor.framework.member_event_sink import MemberEvent, current_member_sink


def _chunk(content: str = "", **metadata: Any) -> SimpleNamespace:
    return SimpleNamespace(content=content, metadata=metadata or {}, tool_calls=[])


class _FakeOrchestrator:
    """Yields a few content chunks; optionally emits member events between them."""

    def __init__(self, *, emit_members: bool = False, flood: int = 0) -> None:
        self._emit_members = emit_members
        self._flood = flood

    async def stream_chat(self, prompt: str) -> AsyncIterator[SimpleNamespace]:
        sink = current_member_sink.get()
        yield _chunk("Hello ")
        if self._emit_members and sink is not None:
            await sink.emit(MemberEvent("member_start", "m1", index=0, formation="sequential"))
        await asyncio.sleep(0)
        yield _chunk("world")
        if self._emit_members and sink is not None:
            await sink.emit(
                MemberEvent("member_completed", "m1", index=0, success=True, formation="sequential")
            )
        if self._flood and sink is not None:
            for i in range(self._flood):
                await sink.emit(MemberEvent("member_start", f"f{i}", index=i))
        await asyncio.sleep(0)


async def _collect(orchestrator: Any) -> List[Any]:
    return [ev async for ev in stream_with_events(orchestrator, "hi")]


def _types(events: List[Any]) -> List[str]:
    return [e.type.value for e in events]


async def test_member_events_interleave_into_stream() -> None:
    events = await _collect(_FakeOrchestrator(emit_members=True))

    # The member lifecycle events surface as CUSTOM events tagged with member_id.
    member_events = [e for e in events if e.type == EventType.CUSTOM]
    kinds = [e.metadata.get("custom_type") for e in member_events]
    assert kinds == ["member_start", "member_completed"]
    assert all(e.member_id == "m1" for e in member_events)
    assert member_events[0].metadata.get("formation") == "sequential"

    # Content still flows and the stream terminates cleanly.
    assert _types(events)[0] == EventType.STREAM_START.value
    assert _types(events)[-1] == EventType.STREAM_END.value
    assert events[-1].success is True
    assert any(e.type == EventType.CONTENT for e in events)


async def test_no_team_is_byte_identical() -> None:
    # With no producer touching the sink, the emitted sequence is exactly the
    # single-agent stream: start, content, end — no CUSTOM member events.
    events = await _collect(_FakeOrchestrator(emit_members=False))
    assert not [e for e in events if e.type == EventType.CUSTOM]
    assert _types(events)[0] == EventType.STREAM_START.value
    assert _types(events)[-1] == EventType.STREAM_END.value
    assert events[-1].success is True
    # The context var is restored to None after the stream completes.
    assert current_member_sink.get() is None


async def test_flood_does_not_deadlock() -> None:
    # Far more member emits than the sink holds must not wedge the turn.
    events = await asyncio.wait_for(_collect(_FakeOrchestrator(emit_members=True, flood=1000)), 5.0)
    assert _types(events)[-1] == EventType.STREAM_END.value
    assert events[-1].success is True
    # Some member events still made it through (bounded, not zero).
    assert any(e.type == EventType.CUSTOM for e in events)


async def test_member_start_precedes_completed() -> None:
    events = await _collect(_FakeOrchestrator(emit_members=True))
    order = [e.metadata.get("custom_type") for e in events if e.type == EventType.CUSTOM]
    assert order.index("member_start") < order.index("member_completed")
