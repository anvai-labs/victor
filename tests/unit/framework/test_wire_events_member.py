"""ADR-023 increment 4: member CUSTOM events serialize to the additive wire shape."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

from victor.framework.wire_events import (
    MEMBER_WIRE_EVENT_TYPES,
    WIRE_EVENT_TYPES,
    WIRE_VERSION,
    to_wire_event,
)


def _custom(
    custom_type: str,
    *,
    member_id: Optional[str] = None,
    content: str = "",
    success: bool = True,
    index: Optional[int] = None,
    formation: Optional[str] = None,
) -> Any:
    metadata = {"custom_type": custom_type, "member_index": index, "formation": formation}
    return SimpleNamespace(
        event_type="custom",
        content=content,
        success=success,
        member_id=member_id,
        metadata={k: v for k, v in metadata.items() if v is not None},
    )


def test_member_start_serializes() -> None:
    wire = to_wire_event(_custom("member_start", member_id="m1", index=0, formation="sequential"))
    assert wire == {
        "v": WIRE_VERSION,
        "event": "member_start",
        "member_id": "m1",
        "index": 0,
        "formation": "sequential",
    }


def test_member_completed_carries_success() -> None:
    wire = to_wire_event(_custom("member_completed", member_id="m1", success=True))
    assert wire["event"] == "member_completed"
    assert wire["success"] is True
    assert wire["member_id"] == "m1"


def test_member_error_carries_failure_and_content() -> None:
    wire = to_wire_event(_custom("member_error", member_id="m2", success=False, content="boom"))
    assert wire["event"] == "member_error"
    assert wire["success"] is False
    assert wire["content"] == "boom"


def test_member_types_are_additive_not_core() -> None:
    # The base six-type contract is unchanged; member types are a separate additive set.
    assert MEMBER_WIRE_EVENT_TYPES == {"member_start", "member_completed", "member_error"}
    assert not (MEMBER_WIRE_EVENT_TYPES & WIRE_EVENT_TYPES)


def test_bare_custom_event_stays_out_of_contract() -> None:
    # A custom event without a member_ sub-type is still None (out of v1 contract).
    bare = SimpleNamespace(event_type="custom", content="x", metadata={"custom_type": "milestone"})
    assert to_wire_event(bare) is None
    no_meta = SimpleNamespace(event_type="custom", content="x", metadata={})
    assert to_wire_event(no_meta) is None
