"""ADR-023 increment 4: member CUSTOM events map to MEMBER_START/END render actions."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

from victor.ui.chat_app.event_mapping import (
    RenderKind,
    map_event,
    map_wire_event,
)


def _event(
    custom_type: str, *, member_id: Optional[str] = None, content: str = "", success: bool = True
) -> Any:
    return SimpleNamespace(
        event_type="custom",
        content=content,
        success=success,
        member_id=member_id,
        metadata={"custom_type": custom_type, "member_id": member_id},
    )


def test_member_start_maps_to_member_start_kind() -> None:
    action = map_event(_event("member_start", member_id="m1"))
    assert action.kind is RenderKind.MEMBER_START
    assert action.member_id == "m1"


def test_member_completed_maps_to_member_end_success() -> None:
    action = map_event(_event("member_completed", member_id="m1"))
    assert action.kind is RenderKind.MEMBER_END
    assert action.success is True
    assert action.member_id == "m1"


def test_member_error_maps_to_member_end_failure() -> None:
    action = map_event(_event("member_error", member_id="m2", content="boom", success=False))
    assert action.kind is RenderKind.MEMBER_END
    assert action.success is False
    assert action.member_id == "m2"


def test_member_awaiting_approval_maps_to_awaiting_kind() -> None:
    # ADR-023 pillar 2b: durable pause → the awaiting-approval lane.
    action = map_event(_event("member_awaiting_approval", member_id="m1", content="run_command"))
    assert action.kind is RenderKind.MEMBER_AWAITING
    assert action.member_id == "m1"
    assert action.text == "run_command"


def test_non_member_custom_event_is_ignored() -> None:
    # A milestone (or any other custom sub-type) still maps to IGNORE, unchanged.
    other = SimpleNamespace(event_type="custom", content="x", metadata={"custom_type": "milestone"})
    assert map_event(other).kind is RenderKind.IGNORE


def test_wire_member_events_map() -> None:
    start = map_wire_event({"v": 1, "event": "member_start", "member_id": "m1"})
    assert start.kind is RenderKind.MEMBER_START and start.member_id == "m1"

    done = map_wire_event({"v": 1, "event": "member_completed", "member_id": "m1", "success": True})
    assert done.kind is RenderKind.MEMBER_END and done.success is True

    err = map_wire_event({"v": 1, "event": "member_error", "member_id": "m2", "success": False})
    assert err.kind is RenderKind.MEMBER_END and err.success is False

    paused = map_wire_event(
        {"v": 1, "event": "member_awaiting_approval", "member_id": "m1", "content": "run_command"}
    )
    assert paused.kind is RenderKind.MEMBER_AWAITING
    assert paused.member_id == "m1"
    assert paused.text == "run_command"
