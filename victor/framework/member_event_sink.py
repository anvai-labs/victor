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

"""Async fan-in of team member lifecycle events into the client stream (ADR-023).

`UnifiedTeamCoordinator` runs as a StateGraph node deep inside a chat turn, so its
members' progress is normally opaque to `VictorClient.stream()` — only the aggregated
result surfaces. This module provides the *seam* (FEP-0028 pillar 3) that lets a running
team push per-member lifecycle events up into the stream funnel
(`victor.framework._internal.stream_with_events`) so the TUI can render per-member lanes.

The handle is carried on a :data:`current_member_sink` ``ContextVar``, which rides the
existing awaited call chain (funnel → orchestrator → chat service → team node → formation)
without threading a parameter through a dozen signatures. When no sink is set
(``default=None`` — every single-agent turn), the producer side is inert and behavior is
byte-identical.

Emission is **bounded and lossy on purpose**: a team member must never block on a slow or
absent stream consumer, so :meth:`MemberEventSink.emit` drops the oldest queued event
rather than await backpressure. Ordering across a *single* (SEQUENTIAL) writer is preserved
by the underlying FIFO queue.
"""

from __future__ import annotations

import asyncio
import contextvars
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Optional

#: Member lifecycle event kinds carried to the client stream. ``member_tool`` and
#: member token streaming are deferred (they need ``SubAgent.stream_execute`` wiring).
MEMBER_START = "member_start"
MEMBER_COMPLETED = "member_completed"
MEMBER_ERROR = "member_error"

#: Default bound on queued-but-undrained member events before drop-oldest kicks in.
DEFAULT_SINK_MAXSIZE = 256


@dataclass(frozen=True)
class MemberEvent:
    """One team member lifecycle event bound for the client stream.

    Attributes:
        kind: One of :data:`MEMBER_START`, :data:`MEMBER_COMPLETED`, :data:`MEMBER_ERROR`.
        member_id: The team member's id (the per-lane key on the UI).
        formation: The active formation name (e.g. ``"sequential"``), for context.
        index: The member's position in the formation, when known.
        content: Short human-readable payload (e.g. an error message).
        success: Whether the member step succeeded (meaningful for completed/error).
        metadata: Additional, forward-compatible detail.
    """

    kind: str
    member_id: str
    formation: Optional[str] = None
    index: Optional[int] = None
    content: str = ""
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


# Sentinel pushed by ``close()`` so the drain loop / merge helper terminates cleanly.
_CLOSE = object()


class MemberEventSink:
    """Ordered, bounded, async fan-in from a running team into the client stream.

    A single sink is created per streamed turn (by ``stream_with_events``) and published on
    :data:`current_member_sink`. Formation strategies push :class:`MemberEvent`s via
    :meth:`emit`; the stream funnel drains them (racing them against the orchestrator's own
    chunk stream) and closes the sink when the primary stream ends.
    """

    def __init__(self, maxsize: int = DEFAULT_SINK_MAXSIZE) -> None:
        self._q: "asyncio.Queue[Any]" = asyncio.Queue(maxsize=maxsize)

    async def emit(self, event: MemberEvent) -> None:
        """Enqueue a member event without ever blocking the emitting team.

        Uses ``put_nowait`` and, on a full queue, drops the oldest event to make room —
        a slow/absent consumer can never wedge team execution. Async-signatured for a
        uniform ``await sink.emit(...)`` call site, but never suspends on backpressure.
        """
        try:
            self._q.put_nowait(event)
        except asyncio.QueueFull:
            try:
                self._q.get_nowait()  # drop oldest
                self._q.put_nowait(event)
            except (asyncio.QueueEmpty, asyncio.QueueFull):  # pragma: no cover - race guard
                pass

    async def close(self) -> None:
        """Signal end-of-stream to the drain loop without ever blocking.

        The close sentinel must always land even when the bounded queue is full (an
        emit burst with no drainer yet), so — like :meth:`emit` — this drops the oldest
        queued event to make room rather than await backpressure. Guarantees the drain
        loop / merge helper terminates.
        """
        while True:
            try:
                self._q.put_nowait(_CLOSE)
                return
            except asyncio.QueueFull:
                try:
                    self._q.get_nowait()  # drop oldest to make room for the sentinel
                except asyncio.QueueEmpty:  # pragma: no cover - race guard
                    pass

    async def get(self) -> Any:
        """Await the next queued item — a :class:`MemberEvent` or the close sentinel."""
        return await self._q.get()

    async def drain(self) -> AsyncIterator[MemberEvent]:
        """Yield queued :class:`MemberEvent`s until :meth:`close` is seen."""
        while True:
            item = await self._q.get()
            if item is _CLOSE:
                return
            yield item


def is_close_sentinel(item: Any) -> bool:
    """True when ``item`` is the sink close sentinel (for the merge helper)."""
    return item is _CLOSE


#: Async-context-propagated sink handle. ``None`` on every single-agent turn (zero overhead);
#: set by the stream funnel for the duration of a streamed turn.
current_member_sink: "contextvars.ContextVar[Optional[MemberEventSink]]" = contextvars.ContextVar(
    "current_member_sink", default=None
)
