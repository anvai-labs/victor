"""ADR-023 increment 4: the member event sink (bounded, lossy, async fan-in)."""

from __future__ import annotations

import asyncio

from victor.framework.member_event_sink import (
    MemberEvent,
    MemberEventSink,
    current_member_sink,
    is_close_sentinel,
)


async def test_emit_drain_preserves_order() -> None:
    sink = MemberEventSink()
    for i in range(3):
        await sink.emit(MemberEvent(kind="member_start", member_id=f"m{i}", index=i))
    await sink.close()

    seen = [ev async for ev in sink.drain()]
    assert [ev.member_id for ev in seen] == ["m0", "m1", "m2"]


async def test_close_terminates_drain() -> None:
    sink = MemberEventSink()
    await sink.close()
    seen = [ev async for ev in sink.drain()]
    assert seen == []


async def test_close_sentinel_via_get() -> None:
    sink = MemberEventSink()
    await sink.emit(MemberEvent(kind="member_completed", member_id="m0", success=True))
    await sink.close()
    first = await sink.get()
    assert isinstance(first, MemberEvent)
    assert is_close_sentinel(await sink.get())


async def test_full_queue_drops_oldest_and_never_blocks() -> None:
    # Tiny queue: emitting far more than capacity must not block or raise.
    sink = MemberEventSink(maxsize=4)
    for i in range(100):
        await asyncio.wait_for(
            sink.emit(MemberEvent(kind="member_start", member_id=f"m{i}", index=i)),
            timeout=1.0,
        )
    await sink.close()

    seen = [ev async for ev in sink.drain()]
    # Bounded: only the most recent events survive (oldest dropped).
    assert len(seen) <= 4
    assert seen[-1].member_id == "m99"


async def test_context_var_defaults_to_none() -> None:
    # Outside a streamed turn, no sink is published (zero-overhead single-agent path).
    assert current_member_sink.get() is None
