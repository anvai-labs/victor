"""The public stream bridge retains structured admission diagnostics."""

from types import SimpleNamespace

from victor.framework._internal import stream_with_events
from victor.framework.member_event_sink import MEMBER_THROTTLED, MemberEvent, current_member_sink
from victor.framework.wire_events import to_wire_event
from victor.ui.chat_app.event_mapping import RenderKind, map_event, map_wire_event


async def test_admission_event_survives_stream_and_wire_and_is_ignored_by_legacy_ui():
    class Orchestrator:
        async def stream_chat(self, prompt):
            await current_member_sink.get().emit(
                MemberEvent(
                    MEMBER_THROTTLED,
                    "queued",
                    metadata={"concurrency_limit": 2, "level": "warning"},
                )
            )
            yield SimpleNamespace(content="done", metadata={}, tool_calls=[])

    events = [event async for event in stream_with_events(Orchestrator(), "task")]
    event = next(event for event in events if event.metadata.get("custom_type") == MEMBER_THROTTLED)
    wire = to_wire_event(event)
    assert wire["concurrency_limit"] == 2
    # Consumer decision: old UI must not mislabel capacity waiting as human approval.
    assert map_event(event).kind == RenderKind.IGNORE
    assert map_wire_event(wire).kind == RenderKind.IGNORE


async def test_conversation_events_survive_bridge_and_wire_without_changing_ui():
    class Orchestrator:
        async def stream_chat(self, prompt):
            await current_member_sink.get().emit(
                MemberEvent("member_spoke", "a", metadata={"transcript_sequence": 3})
            )
            await current_member_sink.get().emit(
                MemberEvent(
                    "member_handoff", "a", metadata={"target_member_id": "b", "sequence": 3}
                )
            )
            yield SimpleNamespace(content="done", metadata={}, tool_calls=[])

    events = [event async for event in stream_with_events(Orchestrator(), "task")]
    selected = [
        event
        for event in events
        if event.metadata.get("custom_type") in {"member_spoke", "member_handoff"}
    ]
    wires = [to_wire_event(event) for event in selected]
    assert wires[0]["transcript_sequence"] == 3
    assert wires[1]["target_member_id"] == "b" and wires[1]["sequence"] == 3
    for event, wire in zip(selected, wires):
        assert map_event(event).kind == RenderKind.IGNORE
        assert map_wire_event(wire).kind == RenderKind.IGNORE
