"""ADR-023: per-member streaming lanes for concurrent formations (PARALLEL / HIERARCHICAL).

The concurrent formations now route each member through the shared
`BaseFormationStrategy._execute_member_with_events` helper, so the same lane events the
SEQUENTIAL formation emits (start / completed / error / awaiting) fire for concurrent members
too — via the coordinator's single `member_event_hook` (the sink ContextVar propagates into
the gather tasks). These tests exercise the formation wiring with a recording hook.
"""

from __future__ import annotations

from typing import Any, List, Tuple

from victor.coordination.formations.base import TeamContext
from victor.coordination.formations.hierarchical import HierarchicalFormation
from victor.coordination.formations.parallel import ParallelFormation
from victor.teams.types import AgentMessage, MemberResult, MessageType


class _FakeAgent:
    def __init__(self, member_id: str, *, fail: bool = False, awaiting: bool = False) -> None:
        self.id = member_id
        self._fail = fail
        self._awaiting = awaiting

    async def execute(self, task: Any, context: Any) -> MemberResult:
        if self._fail:
            raise RuntimeError("boom")
        if self._awaiting:
            return MemberResult(
                member_id=self.id,
                success=False,
                output="",
                metadata={
                    "awaiting_approval": True,
                    "approval_request": {"title": "run_command"},
                },
            )
        return MemberResult(member_id=self.id, success=True, output=f"ok-{self.id}")


def _task() -> AgentMessage:
    return AgentMessage(sender_id="coordinator", content="do it", message_type=MessageType.TASK)


def _context_with_recording_hook() -> Tuple[TeamContext, List[Tuple[str, str]]]:
    ctx = TeamContext(team_id="t", formation="parallel")
    events: List[Tuple[str, str]] = []

    async def _hook(
        kind: str, member_id: str, index: int, *, success: bool = True, content: str = ""
    ) -> None:
        events.append((kind, member_id))

    ctx.member_event_hook = _hook
    return ctx, events


async def test_parallel_emits_per_member_lanes() -> None:
    ctx, events = _context_with_recording_hook()
    agents = [
        _FakeAgent("m0"),
        _FakeAgent("m1", fail=True),
        _FakeAgent("m2", awaiting=True),
    ]
    await ParallelFormation().execute(agents, ctx, _task())

    # Concurrent → order-independent; assert the set of (kind, member) pairs.
    pairs = set(events)
    assert ("member_start", "m0") in pairs
    assert ("member_start", "m1") in pairs
    assert ("member_start", "m2") in pairs
    assert ("member_completed", "m0") in pairs  # success
    assert ("member_error", "m1") in pairs  # raised → normalized failure
    assert ("member_awaiting_approval", "m2") in pairs  # awaiting-approval result


async def test_parallel_without_hook_is_unchanged() -> None:
    # No member_event_hook (no streamed turn) → members just run, results intact.
    ctx = TeamContext(team_id="t", formation="parallel")
    assert getattr(ctx, "member_event_hook", None) is None
    agents = [_FakeAgent("m0"), _FakeAgent("m1")]
    results = await ParallelFormation().execute(agents, ctx, _task())
    assert {r.member_id for r in results} == {"m0", "m1"}
    assert all(r.success for r in results)


async def test_hierarchical_specialist_emits_lane() -> None:
    ctx, events = _context_with_recording_hook()
    result = await HierarchicalFormation()._execute_specialist(_FakeAgent("s1"), _task(), ctx, 0)
    assert result.success is True
    assert ("member_start", "s1") in events
    assert ("member_completed", "s1") in events
