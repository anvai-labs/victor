"""ADR-023: phase-granular durable checkpoint/resume + durable pause for HIERARCHICAL.

HIERARCHICAL runs three phases — supervisor *plan* → concurrent *specialists* → supervisor
*synthesis*. Each completed phase is snapshotted into ``shared_state["__hier__"]`` (persisted by
the existing member checkpoint hook), so a crash resumes at the last completed phase: a completed
plan is restored (not re-run) and drives phase 2; completed specialists are restored; only the
unreached phase(s) execute. The no-delegation fallback path ends at phase 2 (no synthesis). The
specialist wave runs through the shared concurrent runner, so mid-wave cumulative checkpoints give
per-specialist partial resume (a mid-wave crash re-runs only the unfinished specialists — no
replan). Durable pause covers all three phases: the supervisor plan and synthesis pause via the
singular ``__awaiting_approval__`` aggregate (the paused phase re-runs on resume), the specialist
wave via the concurrent multi-pause aggregate (``__awaiting_approvals__`` + one batch pause
checkpoint; completed specialists are skipped on resume).

Driven at the formation layer (fake agents implementing ``execute`` → ``MemberResult``), wiring the
real coordinator checkpoint/pause hooks + resume loader against a ``MemoryCheckpointer`` — so the
checkpoint format is exercised end-to-end while the supervisor/delegation contract stays under test
control.
"""

from __future__ import annotations

from typing import Any, Dict, List

from victor.coordination.formations.base import TeamContext
from victor.coordination.formations.hierarchical import HierarchicalFormation
from victor.framework.graph_checkpoint import MemoryCheckpointer
from victor.teams import UnifiedTeamCoordinator
from victor.teams.types import AgentMessage, MemberResult, MessageType

_THREAD = "t1"
_NODE = "team"


def _approval(member_id: str) -> Dict[str, Any]:
    return {"id": f"req-{member_id}", "tool_name": "run_command", "title": f"cmd-{member_id}"}


def _awaiting(member_id: str) -> MemberResult:
    return MemberResult(
        member_id=member_id,
        success=False,
        output="",
        metadata={"awaiting_approval": True, "approval_request": _approval(member_id)},
    )


class _Supervisor:
    """Delegates one task per specialist on phase 1, synthesizes on phase 3."""

    def __init__(
        self,
        member_id: str,
        specialists: List[str],
        *,
        delegate: bool = True,
        pause_plan: bool = False,
        pause_synth: bool = False,
    ) -> None:
        self.id = member_id
        self._specialists = specialists
        self._delegate = delegate
        self._pause_plan = pause_plan
        self._pause_synth = pause_synth
        self.plan_calls = 0
        self.synth_calls = 0

    async def execute(self, task: AgentMessage, context: Any) -> MemberResult:
        if getattr(task, "message_type", None) == MessageType.RESULT:
            self.synth_calls += 1
            if self._pause_synth:
                return _awaiting(self.id)
            return MemberResult(member_id=self.id, success=True, output="synthesis")
        self.plan_calls += 1
        if self._pause_plan:
            return _awaiting(self.id)
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

    def __init__(self, member_id: str, *, pause: bool = False) -> None:
        self.id = member_id
        self._pause = pause
        self.calls = 0

    async def execute(self, task: AgentMessage, context: Any) -> MemberResult:
        self.calls += 1
        if self._pause:
            return _awaiting(self.id)
        return MemberResult(member_id=self.id, success=True, output=f"ok-{self.id}")


def _task() -> AgentMessage:
    return AgentMessage(sender_id="coordinator", content="go", message_type=MessageType.TASK)


async def _context(cp: Any = None) -> TeamContext:
    """Build a HIERARCHICAL context, wiring the real checkpoint/pause hooks when durable."""
    ctx = TeamContext(
        team_id="t", formation="hierarchical", shared_state={"explicit_supervisor_id": "sup"}
    )
    if cp is not None:
        coord = UnifiedTeamCoordinator(lightweight_mode=True, checkpointer=cp)
        ctx.resume_completed = await coord._load_member_resume(cp, _THREAD, _NODE)
        ctx.checkpoint_hook = coord._make_member_checkpoint_hook(cp, _THREAD, _NODE, "hierarchical")
        ctx.pause_hook = coord._make_member_pause_hook(cp, _THREAD, _NODE, "hierarchical")
        ctx.batch_pause_hook = coord._make_member_batch_pause_hook(
            cp, _THREAD, _NODE, "hierarchical"
        )
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


# ── durable pause: specialist wave (batch aggregate) ──────────────


def _pauses(checkpoints: List[Any]) -> List[Any]:
    return [c for c in checkpoints if c.metadata.get("awaiting_approval")]


async def test_specialist_wave_pauses_on_awaiting_members_and_resumes() -> None:
    cp = MemoryCheckpointer()
    sup = _Supervisor("sup", ["s1", "s2", "s3"])
    s1, s2, s3 = _Specialist("s1", pause=True), _Specialist("s2"), _Specialist("s3", pause=True)
    ctx = await _context(cp)
    results = await HierarchicalFormation().execute([sup, s1, s2, s3], ctx, _task())

    # All three specialists ran once (concurrent wave); two came back awaiting.
    assert s1.calls == 1 and s2.calls == 1 and s3.calls == 1
    # Paused: supervisor plan + the completed specialist only; NO synthesis.
    assert _ids(results) == {"sup", "s2"}
    assert sup.synth_calls == 0
    assert {a["member_id"] for a in ctx.shared_state["__awaiting_approvals__"]} == {"s1", "s3"}

    # Exactly one batch pause checkpoint: completed set excludes the awaiting pair, every pending
    # approval recorded, and the embedded __hier__ snapshot holds ONLY the plan (phase 2 was not
    # snapshotted — the wave did not complete).
    pauses = _pauses(await cp.list(_THREAD))
    assert len(pauses) == 1
    state = pauses[0].state
    assert set(state["completed_member_ids"]) == {"s2"}
    assert {a["member_id"] for a in state["awaiting_approvals"]} == {"s1", "s3"}
    hier = state["shared_state"]["__hier__"]
    assert hier["phase"] == 1 and "specialists" not in hier

    # Resume (approval granted → the pair no longer pauses): the plan is restored (no replan),
    # exactly the paused pair re-runs, the completed specialist is skipped, then synthesis runs.
    sup2 = _Supervisor("sup", ["s1", "s2", "s3"])
    r1, r2, r3 = _Specialist("s1"), _Specialist("s2"), _Specialist("s3")
    resumed = await _run([sup2, r1, r2, r3], cp)
    assert sup2.plan_calls == 0  # restored plan — no replan
    assert r1.calls == 1 and r3.calls == 1  # paused pair re-runs
    assert r2.calls == 0  # completed specialist skipped
    assert sup2.synth_calls == 1  # synthesis now runs
    assert all(r.success for r in resumed) and _ids(resumed) == {"sup", "s1", "s2", "s3"}
    assert resumed[0].output == "synthesis"


# ── durable pause: supervisor plan (phase 1) ──────────────────────


async def test_supervisor_plan_pause_reruns_plan_on_resume() -> None:
    cp = MemoryCheckpointer()
    sup = _Supervisor("sup", ["s1", "s2"], pause_plan=True)
    s1, s2 = _Specialist("s1"), _Specialist("s2")
    ctx = await _context(cp)
    results = await HierarchicalFormation().execute([sup, s1, s2], ctx, _task())

    # Paused before any phase completed: no results, no specialists, no synthesis.
    assert results == []
    assert s1.calls == 0 and s2.calls == 0 and sup.synth_calls == 0
    assert ctx.shared_state["__awaiting_approval__"]["member_id"] == "sup"
    pauses = _pauses(await cp.list(_THREAD))
    assert len(pauses) == 1
    assert pauses[0].state["completed_member_ids"] == []
    assert pauses[0].state["approval_request"] == _approval("sup")
    # The awaiting plan was NOT snapshotted — resume re-executes it.
    assert "__hier__" not in (pauses[0].state["shared_state"] or {})

    sup2 = _Supervisor("sup", ["s1", "s2"])
    r1, r2 = _Specialist("s1"), _Specialist("s2")
    resumed = await _run([sup2, r1, r2], cp)
    assert sup2.plan_calls == 1  # plan re-runs on resume
    assert r1.calls == 1 and r2.calls == 1
    assert sup2.synth_calls == 1
    assert all(r.success for r in resumed) and _ids(resumed) == {"sup", "s1", "s2"}


# ── durable pause: supervisor synthesis (phase 3) ─────────────────


async def test_synthesis_pause_restores_phases_and_reruns_synthesis_only() -> None:
    cp = MemoryCheckpointer()
    sup = _Supervisor("sup", ["s1", "s2"], pause_synth=True)
    s1, s2 = _Specialist("s1"), _Specialist("s2")
    ctx = await _context(cp)
    results = await HierarchicalFormation().execute([sup, s1, s2], ctx, _task())

    assert sup.plan_calls == 1 and sup.synth_calls == 1
    assert results[0].output == "plan"  # awaiting synthesis did NOT replace the supervisor slot
    assert ctx.shared_state["__awaiting_approval__"]["member_id"] == "sup"
    pauses = _pauses(await cp.list(_THREAD))
    assert len(pauses) == 1
    hier = pauses[0].state["shared_state"]["__hier__"]
    assert hier["phase"] == 2  # phases 1–2 embedded in the pause checkpoint
    assert "synthesis" not in hier  # the awaiting synthesis was NOT snapshotted

    sup2 = _Supervisor("sup", ["s1", "s2"])
    r1, r2 = _Specialist("s1"), _Specialist("s2")
    resumed = await _run([sup2, r1, r2], cp)
    assert sup2.plan_calls == 0 and r1.calls == 0 and r2.calls == 0  # phases 1–2 restored
    assert sup2.synth_calls == 1  # only the synthesis re-runs
    assert all(r.success for r in resumed)
    assert resumed[0].output == "synthesis"


# ── per-specialist partial resume (mid-wave crash) ────────────────


async def test_midwave_crash_resume_skips_completed_specialist_without_replan() -> None:
    # A crash mid specialist wave leaves the phase-1 plan snapshot + a mid-wave cumulative
    # checkpoint for the specialist that finished. Simulated by running once and keeping only a
    # mid-wave checkpoint (its __hier__ still holds just the plan; its completed set holds s1).
    cp = MemoryCheckpointer()
    await _run([_Supervisor("sup", ["s1", "s2"]), _Specialist("s1"), _Specialist("s2")], cp)
    midwave = [
        c
        for c in _hiers(await cp.list(_THREAD))
        if _phase_of(c) == 1 and set(c.state.get("completed_member_ids") or []) == {"s1"}
    ]
    assert midwave, "expected a mid-wave cumulative checkpoint for s1"
    cp2 = MemoryCheckpointer()
    await cp2.save(midwave[0])

    sup2 = _Supervisor("sup", ["s1", "s2"])
    r1, r2 = _Specialist("s1"), _Specialist("s2")
    resumed = await _run([sup2, r1, r2], cp2)
    assert sup2.plan_calls == 0  # NO replan — the mid-wave checkpoint embeds the plan
    assert r1.calls == 0  # completed specialist skipped
    assert r2.calls == 1  # only the unfinished specialist re-runs
    assert sup2.synth_calls == 1
    assert all(r.success for r in resumed) and _ids(resumed) == {"sup", "s1", "s2"}


# ── opt-out ───────────────────────────────────────────────────────


async def test_no_checkpointer_awaiting_specialist_does_not_pause() -> None:
    # Without hooks, awaiting-approval is an inert non-success result: the wave completes, no
    # aggregate is published, and synthesis still runs — byte-identical to pre-pause behavior.
    sup = _Supervisor("sup", ["s1", "s2"])
    s1, s2 = _Specialist("s1", pause=True), _Specialist("s2")
    ctx = await _context()
    results = await HierarchicalFormation().execute([sup, s1, s2], ctx, _task())
    assert "__awaiting_approvals__" not in ctx.shared_state
    assert "__awaiting_approval__" not in ctx.shared_state
    assert "__hier__" not in ctx.shared_state
    assert sup.synth_calls == 1
    assert _ids(results) == {"sup", "s1", "s2"}


async def test_no_checkpointer_is_unchanged() -> None:
    sup = _Supervisor("sup", ["s1", "s2"])
    s1, s2 = _Specialist("s1"), _Specialist("s2")
    results = await _run([sup, s1, s2])
    assert sup.plan_calls == 1 and sup.synth_calls == 1
    assert s1.calls == 1 and s2.calls == 1
    assert all(r.success for r in results) and _ids(results) == {"sup", "s1", "s2"}
    assert results[0].output == "synthesis"
