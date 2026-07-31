"""ADR-023: durable-pause arming is gated to formations that implement it.

#740 armed `current_member_durable_pause_enabled` for any checkpointer-backed team, but only
SEQUENTIAL stops-and-checkpoints on an awaiting member. Arming a concurrent formation (which
can't handle the pause) would make a member ASK silently abort the gated tool. The coordinator
now arms only when `strategy.supports_durable_pause()` — so concurrent teams keep slice-2a
inline approval. These tests probe the armed ContextVar during member execution.
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


async def test_parallel_does_not_arm_durable_pause() -> None:
    # PARALLEL cannot stop-and-checkpoint → must NOT arm, even with a checkpointer.
    probe = _ArmingProbe("m0")
    await _coordinator([probe], TeamFormation.PARALLEL, MemoryCheckpointer()).execute_task(
        "do it", {"thread_id": "t1"}
    )
    assert probe.seen_armed is None
