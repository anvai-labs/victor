"""ADR-023: round/iteration-granular durable checkpoint/resume for the iterative formations.

CONSENSUS runs sequential rounds (each round's task is built from the previous round's results);
REFLECTION runs sequential generator→critic iterations (each refines from the previous critique).
Both snapshot their loop state into ``shared_state`` after each round/iteration (persisted by the
existing member checkpoint hook), so a crash resumes at the next unfinished round/iteration —
completed ones are restored, not re-run.

Driven at the formation layer, wiring the real coordinator checkpoint hook + resume loader against
a ``MemoryCheckpointer`` so the checkpoint format is exercised end-to-end.
"""

from __future__ import annotations

from typing import Any, List, Optional

from victor.coordination.formations.base import TeamContext
from victor.coordination.formations.consensus import ConsensusFormation
from victor.coordination.formations.reflection import ReflectionFormation
from victor.framework.graph_checkpoint import MemoryCheckpointer
from victor.teams import UnifiedTeamCoordinator
from victor.teams.types import AgentMessage, MemberResult, MessageType

_THREAD = "t1"
_NODE = "team"


async def _durable_context(cp: Any, extra: Optional[dict] = None) -> TeamContext:
    ctx = TeamContext(team_id="t", formation="team", shared_state=dict(extra or {}))
    coord = UnifiedTeamCoordinator(lightweight_mode=True, checkpointer=cp)
    ctx.resume_completed = await coord._load_member_resume(cp, _THREAD, _NODE)
    ctx.checkpoint_hook = coord._make_member_checkpoint_hook(cp, _THREAD, _NODE, "team")
    return ctx


def _task() -> AgentMessage:
    return AgentMessage(sender_id="coordinator", content="solve it", message_type=MessageType.TASK)


# ══════════════════════════ CONSENSUS ══════════════════════════


class _ConsensusAgent:
    """Round agent recording which rounds it ran (via the task's consensus_round tag)."""

    def __init__(self, member_id: str, *, succeed: bool = True) -> None:
        self.id = member_id
        self._succeed = succeed
        self.rounds_seen: List[int] = []

    async def execute(self, task: AgentMessage, context: Any) -> MemberResult:
        rnd = int(task.data.get("consensus_round", 0)) if isinstance(task.data, dict) else 0
        self.rounds_seen.append(rnd)
        return MemberResult(member_id=self.id, success=self._succeed, output=f"{self.id}-r{rnd}")


def _agents() -> List[_ConsensusAgent]:
    # A dissenter that never succeeds keeps consensus unreachable (ConsensusFormation treats
    # all-success as consensus), so every run deterministically exhausts all rounds.
    return [_ConsensusAgent("a"), _ConsensusAgent("d", succeed=False)]


def _consensus(rounds: int) -> ConsensusFormation:
    return ConsensusFormation(max_rounds=rounds, agreement_threshold=1.1)


def _consensus_snaps(checkpoints: List[Any]) -> List[Any]:
    return [c for c in checkpoints if "__consensus__" in (c.state.get("shared_state") or {})]


async def test_consensus_checkpoints_each_round() -> None:
    cp = MemoryCheckpointer()
    a, d = _agents()
    await _consensus(3).execute([a, d], await _durable_context(cp), _task())

    assert a.rounds_seen == [0, 1, 2]  # all three rounds ran (consensus never reached)
    snaps = _consensus_snaps(await cp.list(_THREAD))
    assert {int(c.state["shared_state"]["__consensus__"]["round_done"]) for c in snaps} == {1, 2, 3}
    latest = max(snaps, key=lambda c: c.timestamp)
    assert latest.state["shared_state"]["__consensus__"]["done"] is True


async def test_consensus_resume_skips_completed_rounds() -> None:
    cp = MemoryCheckpointer()
    await _consensus(3).execute(list(_agents()), await _durable_context(cp), _task())

    # Keep only the round_done=1 snapshot → simulate a crash after round 1 (round_num 0).
    r1 = [
        c
        for c in _consensus_snaps(await cp.list(_THREAD))
        if int(c.state["shared_state"]["__consensus__"]["round_done"]) == 1
    ]
    assert r1
    cp2 = MemoryCheckpointer()
    await cp2.save(r1[0])

    a2, d2 = _agents()
    await _consensus(3).execute([a2, d2], await _durable_context(cp2), _task())

    # Round 0 was restored; only the remaining rounds (consensus_round 1, 2) re-run.
    assert a2.rounds_seen == [1, 2] and d2.rounds_seen == [1, 2]


async def test_consensus_no_checkpointer_is_unchanged() -> None:
    a, d = _agents()
    ctx = TeamContext(team_id="t", formation="team", shared_state={})
    await _consensus(2).execute([a, d], ctx, _task())
    assert a.rounds_seen == [0, 1] and d.rounds_seen == [0, 1]  # both rounds ran


# ══════════════════════════ REFLECTION ══════════════════════════


class _Generator:
    """Generator recording each call; output encodes the iteration count."""

    def __init__(self) -> None:
        self.id = "generator"
        self.calls = 0

    async def execute(self, prompt: str, context: Any = None) -> str:
        self.calls += 1
        return f"draft-{self.calls}"


class _Critic:
    """Critic that stays unsatisfied for the first ``satisfy_at - 1`` calls."""

    def __init__(self, satisfy_at: int = 999) -> None:
        self.id = "critic"
        self.calls = 0
        self._satisfy_at = satisfy_at

    async def execute(self, prompt: str, context: Any = None) -> str:
        self.calls += 1
        return "VERDICT: satisfied" if self.calls >= self._satisfy_at else "VERDICT: needs work"


async def _reflection_context(cp: Any, gen: Any, crit: Any, max_iter: int) -> TeamContext:
    ctx = await _durable_context(cp, {"reflection_max_iterations": max_iter})
    ctx.set("generator", gen)
    ctx.set("critic", crit)
    return ctx


async def test_reflection_checkpoints_each_iteration() -> None:
    cp = MemoryCheckpointer()
    gen, crit = _Generator(), _Critic(satisfy_at=999)  # never satisfied → runs all iterations
    ctx = await _reflection_context(cp, gen, crit, 3)
    results = await ReflectionFormation().execute([], ctx, _task())
    assert results[0].metadata["iterations"] == 3

    snaps = [
        c for c in await cp.list(_THREAD) if "__reflection__" in (c.state.get("shared_state") or {})
    ]
    iters_done = {int(c.state["shared_state"]["__reflection__"]["iter_done"]) for c in snaps}
    assert {1, 2, 3}.issubset(iters_done)
    latest = max(snaps, key=lambda c: c.timestamp)
    assert latest.state["shared_state"]["__reflection__"]["done"] is True


async def test_reflection_resume_skips_completed_iterations() -> None:
    cp = MemoryCheckpointer()
    gen1, crit1 = _Generator(), _Critic(satisfy_at=999)
    await ReflectionFormation().execute([], await _reflection_context(cp, gen1, crit1, 3), _task())

    # Keep only the iteration-1 snapshot → simulate a crash after iteration 1.
    snaps = [
        c for c in await cp.list(_THREAD) if "__reflection__" in (c.state.get("shared_state") or {})
    ]
    i1 = [c for c in snaps if int(c.state["shared_state"]["__reflection__"]["iter_done"]) == 1]
    assert i1
    cp2 = MemoryCheckpointer()
    await cp2.save(i1[0])

    gen2, crit2 = _Generator(), _Critic(satisfy_at=999)
    results = await ReflectionFormation().execute(
        [], await _reflection_context(cp2, gen2, crit2, 3), _task()
    )
    # Iteration 1 restored; only iterations 2 and 3 re-run (two more gen/critic calls).
    assert gen2.calls == 2 and crit2.calls == 2
    assert results[0].metadata["iterations"] == 3


async def test_reflection_terminal_resume_returns_without_rerun() -> None:
    cp = MemoryCheckpointer()
    gen1, crit1 = _Generator(), _Critic(satisfy_at=2)  # satisfied on iteration 2
    first = await ReflectionFormation().execute(
        [], await _reflection_context(cp, gen1, crit1, 5), _task()
    )
    assert first[0].metadata["satisfied"] is True
    assert first[0].metadata["iterations"] == 2

    # Resume on the same (complete) checkpoint → nothing re-runs; the aggregate is rebuilt.
    gen2, crit2 = _Generator(), _Critic(satisfy_at=2)
    resumed = await ReflectionFormation().execute(
        [], await _reflection_context(cp, gen2, crit2, 5), _task()
    )
    assert gen2.calls == 0 and crit2.calls == 0
    assert resumed[0].metadata["satisfied"] is True
    assert resumed[0].metadata["iterations"] == 2


async def test_reflection_no_checkpointer_is_unchanged() -> None:
    gen, crit = _Generator(), _Critic(satisfy_at=999)
    ctx = TeamContext(team_id="t", formation="team", shared_state={"reflection_max_iterations": 2})
    ctx.set("generator", gen)
    ctx.set("critic", crit)
    results = await ReflectionFormation().execute([], ctx, _task())
    assert gen.calls == 2 and crit.calls == 2
    assert results[0].metadata["iterations"] == 2
