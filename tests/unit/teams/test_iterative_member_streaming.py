"""ADR-023: per-member streaming lanes for the iterative formations (CONSENSUS, REFLECTION).

This completes streaming-lane coverage across the whole formation taxonomy. CONSENSUS reuses
the shared `_execute_member_with_events` helper (its round members use `agent.execute(task,
ctx)`); REFLECTION emits inline via the same `member_event_hook` (its generator/critic use a
different execute signature and produce one aggregate result, so the helper doesn't fit).
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from victor.coordination.formations.base import TeamContext
from victor.coordination.formations.consensus import ConsensusFormation
from victor.coordination.formations.reflection import ReflectionFormation
from victor.teams.types import AgentMessage, MemberResult, MessageType


def _task() -> AgentMessage:
    return AgentMessage(sender_id="coordinator", content="go", message_type=MessageType.TASK)


def _recording_context() -> Tuple[TeamContext, List[Tuple[str, str]]]:
    ctx = TeamContext(team_id="t", formation="iterative")
    events: List[Tuple[str, str]] = []

    async def _hook(kind: str, member_id: str, index: int, **kw: Any) -> None:
        events.append((kind, member_id))

    ctx.member_event_hook = _hook
    return ctx, events


# ── CONSENSUS (helper reuse) ───────────────────────────────────────


class _FakeAgent:
    def __init__(self, member_id: str) -> None:
        self.id = member_id

    async def execute(self, task: Any, context: Any) -> MemberResult:
        return MemberResult(member_id=self.id, success=True, output=f"ok-{self.id}")


async def test_consensus_emits_per_member_lanes() -> None:
    ctx, events = _recording_context()
    await ConsensusFormation(max_rounds=1).execute(
        [_FakeAgent("a0"), _FakeAgent("a1")], ctx, _task()
    )
    pairs = set(events)
    assert ("member_start", "a0") in pairs and ("member_completed", "a0") in pairs
    assert ("member_start", "a1") in pairs and ("member_completed", "a1") in pairs


# ── REFLECTION (inline emit) ───────────────────────────────────────


class _FakeGenCritic:
    def __init__(self, member_id: str, response: str) -> None:
        self.id = member_id
        self._response = response

    async def execute(self, content: str, context: Optional[Any] = None) -> str:
        return self._response


async def test_reflection_emits_generator_and_critic_lanes() -> None:
    ctx, events = _recording_context()
    generator = _FakeGenCritic("gen", "a solution")
    critic = _FakeGenCritic("crit", "Looks good. VERDICT: SATISFIED")
    ctx.set("generator", generator)
    ctx.set("critic", critic)
    ctx.shared_state["reflection_max_iterations"] = 1

    await ReflectionFormation().execute([generator, critic], ctx, _task())

    pairs = set(events)
    assert ("member_start", "gen") in pairs and ("member_completed", "gen") in pairs
    assert ("member_start", "crit") in pairs and ("member_completed", "crit") in pairs


async def test_no_hook_is_unchanged() -> None:
    # No member_event_hook → both formations run normally, emit nothing.
    ctx = TeamContext(team_id="t", formation="iterative")
    results = await ConsensusFormation(max_rounds=1).execute(
        [_FakeAgent("a0"), _FakeAgent("a1")], ctx, _task()
    )
    assert {r.member_id for r in results} == {"a0", "a1"}
