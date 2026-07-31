"""ADR-023 increment 4: SEQUENTIAL formation emits member lifecycle events."""

from __future__ import annotations

from typing import Any, List, Tuple

from victor.coordination.formations.base import TeamContext
from victor.coordination.formations.sequential import SequentialFormation
from victor.teams.types import AgentMessage, MemberResult, MessageType


class _FakeAgent:
    def __init__(self, member_id: str, *, fail: bool = False) -> None:
        self.id = member_id
        self._fail = fail

    async def execute(self, task: Any, context: Any) -> MemberResult:
        if self._fail:
            raise RuntimeError("boom")
        return MemberResult(member_id=self.id, success=True, output=f"out-{self.id}")


def _task() -> AgentMessage:
    return AgentMessage(sender_id="coordinator", content="do it", message_type=MessageType.TASK)


def _context_with_hook() -> Tuple[TeamContext, List[Tuple[str, str, int, bool]]]:
    ctx = TeamContext(team_id="t", formation="sequential")
    calls: List[Tuple[str, str, int, bool]] = []

    async def _hook(
        kind: str, member_id: str, index: int, *, success: bool = True, content: str = ""
    ) -> None:
        calls.append((kind, member_id, index, success))

    ctx.member_event_hook = _hook
    return ctx, calls


async def test_emits_start_then_completed_per_member() -> None:
    ctx, calls = _context_with_hook()
    agents = [_FakeAgent("m0"), _FakeAgent("m1")]
    await SequentialFormation().execute(agents, ctx, _task())

    assert calls == [
        ("member_start", "m0", 0, True),
        ("member_completed", "m0", 0, True),
        ("member_start", "m1", 1, True),
        ("member_completed", "m1", 1, True),
    ]


async def test_failure_emits_member_error() -> None:
    ctx, calls = _context_with_hook()
    agents = [_FakeAgent("m0", fail=True), _FakeAgent("m1")]
    await SequentialFormation().execute(agents, ctx, _task())

    kinds = [(c[0], c[1]) for c in calls]
    assert ("member_start", "m0") in kinds
    assert ("member_error", "m0") in kinds  # failed member reports error, run continues
    assert ("member_completed", "m1") in kinds


async def test_no_hook_is_unchanged() -> None:
    # Without a member_event_hook the formation runs normally and emits nothing.
    ctx = TeamContext(team_id="t", formation="sequential")
    assert getattr(ctx, "member_event_hook", None) is None
    agents = [_FakeAgent("m0"), _FakeAgent("m1")]
    results = await SequentialFormation().execute(agents, ctx, _task())
    assert [r.member_id for r in results] == ["m0", "m1"]
    assert all(r.success for r in results)
