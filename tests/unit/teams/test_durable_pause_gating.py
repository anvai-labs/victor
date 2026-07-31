"""ADR-023: durable-pause arming is gated to formations that implement it.

#740 armed `current_member_durable_pause_enabled` for any checkpointer-backed team, but only
formations that stop-and-checkpoint (or, for concurrent waves, collect-and-checkpoint) on an
awaiting member can handle the pause. Arming one that can't would make a member ASK silently
abort the gated tool. The coordinator arms only when `strategy.supports_durable_pause()` — now
true for SEQUENTIAL, PIPELINE, and PARALLEL (via the concurrent multi-pause aggregate), still
false for HIERARCHICAL/CONSENSUS/REFLECTION. These tests probe the armed ContextVar during
member execution and the gating predicate itself.
"""

from __future__ import annotations

from typing import Any, List

from victor.framework.graph_checkpoint import MemoryCheckpointer
from victor.teams import TeamFormation, UnifiedTeamCoordinator


class _ArmingProbe:
    def __init__(self, member_id: str) -> None:
        self.id = member_id
        self.seen_armed: Any = "unset"

    async def execute_task(self, *args: Any, **kwargs: Any) -> dict:
        from victor.agent.member_approval_context import current_member_durable_pause_enabled

        self.seen_armed = current_member_durable_pause_enabled.get()
        return {"output": "ok", "success": True}

    async def receive_message(self, *args: Any, **kwargs: Any) -> None:
        return None


def _coordinator(
    members: List[_ArmingProbe], formation: TeamFormation, checkpointer: Any = None
) -> UnifiedTeamCoordinator:
    coord = UnifiedTeamCoordinator(lightweight_mode=True, checkpointer=checkpointer)
    for member in members:
        coord.add_member(member)
    coord.set_formation(formation)
    return coord


async def test_sequential_arms_durable_pause() -> None:
    probe = _ArmingProbe("m0")
    await _coordinator([probe], TeamFormation.SEQUENTIAL, MemoryCheckpointer()).execute_task(
        "do it", {"thread_id": "t1"}
    )
    assert probe.seen_armed is True  # SEQUENTIAL supports durable pause → armed


async def test_parallel_arms_durable_pause() -> None:
    # PARALLEL now implements durable pause via the concurrent multi-pause aggregate → armed.
    probe = _ArmingProbe("m0")
    await _coordinator([probe], TeamFormation.PARALLEL, MemoryCheckpointer()).execute_task(
        "do it", {"thread_id": "t1"}
    )
    assert probe.seen_armed is True


def test_gating_predicate_matches_implemented_formations() -> None:
    # The arming gate reads supports_durable_pause(): true only for formations that can actually
    # handle an awaiting member (stop/collect + checkpoint). HIERARCHICAL is not there yet.
    from victor.coordination.formations.hierarchical import HierarchicalFormation
    from victor.coordination.formations.parallel import ParallelFormation
    from victor.coordination.formations.pipeline import PipelineFormation
    from victor.coordination.formations.sequential import SequentialFormation

    assert SequentialFormation().supports_durable_pause() is True
    assert PipelineFormation().supports_durable_pause() is True
    assert ParallelFormation().supports_durable_pause() is True
    assert HierarchicalFormation().supports_durable_pause() is False
