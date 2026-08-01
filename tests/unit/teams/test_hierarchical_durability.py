"""ADR-023: phase-granular durable checkpoint/resume for the HIERARCHICAL formation.

HIERARCHICAL runs three phases — supervisor *plan* → concurrent *specialists* → supervisor
*synthesis*. Each completed phase is snapshotted into ``shared_state["__hier__"]`` (persisted by
the existing member checkpoint hook), so a crash resumes at the last completed phase: a completed
plan is restored (not re-run) and drives phase 2; completed specialists are restored; only the
unreached phase(s) execute. The no-delegation fallback path ends at phase 2 (no synthesis).

Driven at the formation layer (fake agents implementing ``execute`` → ``MemberResult``), wiring the
real coordinator checkpoint hook + resume loader against a ``MemoryCheckpointer`` — so the checkpoint
format is exercised end-to-end while the supervisor/delegation contract stays under test control.
"""

from __future__ import annotations

from typing import Any, List

from victor.coordination.formations.base import TeamContext
from victor.coordination.formations.hierarchical import HierarchicalFormation
from victor.framework.graph_checkpoint import MemoryCheckpointer
from victor.teams import UnifiedTeamCoordinator
from victor.teams.types import AgentMessage, MemberResult, MessageType

_THREAD = "t1"
_NODE = "team"


class _Supervisor:
    """Delegates one task per specialist on phase 1, synthesizes on phase 3."""

    def __init__(self, member_id: str, specialists: List[str], *, delegate: bool = True) -> None:
        self.id = member_id
        self._specialists = specialists
        self._delegate = delegate
        self.plan_calls = 0
        self.synth_calls = 0

    async def execute(self, task: AgentMessage, context: Any) -> MemberResult:
        if getattr(task, "message_type", None) == MessageType.RESULT:
            self.synth_calls += 1
            return MemberResult(member_id=self.id, success=True, output="synthesis")
        self.plan_calls += 1
        if not self._delegate:
            return MemberResult(member_id=self.id, success=True, output="plan-no-delegation")
        delegated = [
            AgentMessage(
                sender_id=self.id,
                message_type=MessageType.TASK,
                recipient_id=s,
                content=f"do-{s}",
            )
            for s in self._specialists
        ]
        return MemberResult(
            member_id=self.id,
            success=True,
            output="plan",
            metadata={"delegated_tasks": delegated},
        )


class _Specialist:
    """Specialist recording how often it ran (to prove resume skips completed phases)."""

    def __init__(self, member_id: str) -> None:
        self.id = member_id
        self.calls = 0

    async def execute(self, task: AgentMessage, context: Any) -> MemberResult:
        self.calls += 1
        return MemberResult(member_id=self.id, success=True, output=f"ok-{self.id}")


def _task() -> AgentMessage:
    return AgentMessage(sender_id="coordinator", content="go", message_type=MessageType.TASK)


async def _context(cp: Any = None) -> TeamContext:
    """Build a HIERARCHICAL context, wiring the real checkpoint hook + resume loader when durable."""
    ctx = TeamContext(
        team_id="t", formation="hierarchical", shared_state={"explicit_supervisor_id": "sup"}
    )
    if cp is not None:
        coord = UnifiedTeamCoordinator(lightweight_mode=True, checkpointer=cp)
        ctx.resume_completed = await coord._load_member_resume(cp, _THREAD, _NODE)
        ctx.checkpoint_hook = coord._make_member_checkpoint_hook(cp, _THREAD, _NODE, "hierarchical")
    return ctx


async def _run(agents: List[Any], cp: Any = None) -> List[MemberResult]:
    return await HierarchicalFormation().execute(agents, await _context(cp), _task())


def _hiers(checkpoints: List[Any]) -> List[Any]:
    return [c for c in checkpoints if "__hier__" in (c.state.get("shared_state") or {})]


def _phase_of(checkpoint: Any) -> int:
    return int((checkpoint.state["shared_state"]["__hier__"]).get("phase", 0))


def _ids(results: List[MemberResult]) -> set:
    return {r.member_id for r in results}


# ── checkpoint: a phase snapshot per phase ────────────────────────


async def test_hierarchical_checkpoints_each_phase() -> None:
    cp = MemoryCheckpointer()
    results = await _run(
        [_Supervisor("sup", ["s1", "s2"]), _Specialist("s1"), _Specialist("s2")], cp
    )
    assert all(r.success for r in results)
    assert _ids(results) == {"sup", "s1", "s2"}

    hiers = _hiers(await cp.list(_THREAD))
    assert {_phase_of(c) for c in hiers} == {1, 2, 3}  # plan, specialists, synthesis
    latest = max(hiers, key=lambda c: c.timestamp)
    snap = latest.state["shared_state"]["__hier__"]
    assert snap["phase"] == 3
    assert snap["plan"]["member_id"] == "sup"
    assert {r["member_id"] for r in snap["specialists"]} == {"s1", "s2"}
    assert snap["synthesis"]["output"] == "synthesis"


# ── resume: restore up to the last completed phase ────────────────


async def test_resume_after_full_run_reruns_nothing() -> None:
    cp = MemoryCheckpointer()
    await _run([_Supervisor("sup", ["s1", "s2"]), _Specialist("s1"), _Specialist("s2")], cp)

    sup2 = _Supervisor("sup", ["s1", "s2"])
    s1, s2 = _Specialist("s1"), _Specialist("s2")
    results = await _run([sup2, s1, s2], cp)

    assert sup2.plan_calls == 0 and sup2.synth_calls == 0  # every phase restored
    assert s1.calls == 0 and s2.calls == 0
    assert all(r.success for r in results) and _ids(results) == {"sup", "s1", "s2"}
    assert results[0].output == "synthesis"  # restored synthesis is the supervisor slot


async def test_resume_after_plan_reruns_specialists_then_synthesizes() -> None:
    # A crash *after* the plan but *before* the specialist wave completes: the latest snapshot is
    # the phase-1 plan. Simulated by running once, then keeping only the phase-1 snapshot (a
    # mid-wave crash never persists phases 2/3).
    cp = MemoryCheckpointer()
    await _run([_Supervisor("sup", ["s1", "s2"]), _Specialist("s1"), _Specialist("s2")], cp)
    plan_only = [c for c in _hiers(await cp.list(_THREAD)) if _phase_of(c) == 1]
    assert plan_only, "expected a phase-1 plan snapshot"
    cp2 = MemoryCheckpointer()
    await cp2.save(plan_only[0])

    sup2 = _Supervisor("sup", ["s1", "s2"])
    s1, s2 = _Specialist("s1"), _Specialist("s2")
    results = await _run([sup2, s1, s2], cp2)

    assert sup2.plan_calls == 0  # plan restored, not re-run
    assert s1.calls == 1 and s2.calls == 1  # specialists re-run (mid-wave crash → replay wave)
    assert sup2.synth_calls == 1  # synthesis then runs
    assert all(r.success for r in results) and _ids(results) == {"sup", "s1", "s2"}


# ── fallback (no delegation) path ─────────────────────────────────


async def test_fallback_path_checkpoints_two_phases_and_resumes() -> None:
    cp = MemoryCheckpointer()
    results = await _run(
        [_Supervisor("sup", ["s1", "s2"], delegate=False), _Specialist("s1"), _Specialist("s2")], cp
    )
    assert all(r.success for r in results) and _ids(results) == {"sup", "s1", "s2"}
    # No synthesis on the fallback path → only phases 1 and 2 snapshotted.
    assert {_phase_of(c) for c in _hiers(await cp.list(_THREAD))} == {1, 2}

    sup2 = _Supervisor("sup", ["s1", "s2"], delegate=False)
    s1, s2 = _Specialist("s1"), _Specialist("s2")
    resumed = await _run([sup2, s1, s2], cp)
    assert sup2.plan_calls == 0 and s1.calls == 0 and s2.calls == 0
    assert sup2.synth_calls == 0  # fallback never synthesizes
    assert all(r.success for r in resumed)


# ── opt-out ───────────────────────────────────────────────────────


async def test_no_checkpointer_is_unchanged() -> None:
    sup = _Supervisor("sup", ["s1", "s2"])
    s1, s2 = _Specialist("s1"), _Specialist("s2")
    results = await _run([sup, s1, s2])
    assert sup.plan_calls == 1 and sup.synth_calls == 1
    assert s1.calls == 1 and s2.calls == 1
    assert all(r.success for r in results) and _ids(results) == {"sup", "s1", "s2"}
    assert results[0].output == "synthesis"
