# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""UX P1: the versioned JSON wire contract for agent events."""

from __future__ import annotations

import json
from types import SimpleNamespace

from victor.framework.wire_events import (
    MAX_RESULT_CHARS,
    WIRE_EVENT_TYPES,
    WIRE_VERSION,
    encode_sse,
    stream_wire_events,
    to_wire_event,
)


def _event(event_type: str, **kwargs) -> SimpleNamespace:
    defaults = {
        "content": None,
        "tool_name": None,
        "arguments": None,
        "result": None,
        "success": True,
    }
    defaults.update(kwargs)
    return SimpleNamespace(event_type=event_type, **defaults)


class TestToWireEvent:
    def test_all_six_contract_types_serialize(self):
        wires = [
            to_wire_event(_event("thinking", content="hmm")),
            to_wire_event(_event("tool_call", tool_name="read", arguments={"path": "a.py"})),
            to_wire_event(_event("tool_result", tool_name="read", result="ok")),
            to_wire_event(_event("content", content="hello")),
            to_wire_event(_event("error", content="boom")),
            to_wire_event(_event("stream_end")),
        ]
        assert all(w is not None for w in wires)
        assert {w["event"] for w in wires} == WIRE_EVENT_TYPES
        assert all(w["v"] == WIRE_VERSION for w in wires)

    def test_tool_error_maps_to_failed_tool_result(self):
        wire = to_wire_event(_event("tool_error", tool_name="shell", result="exit 1"))
        assert wire["event"] == "tool_result"
        assert wire["success"] is False

    def test_out_of_contract_types_return_none(self):
        for event_type in ("tool_progress", "stage_change", "milestone", "custom", ""):
            assert to_wire_event(_event(event_type, content="x")) is None

    def test_empty_content_events_are_dropped(self):
        assert to_wire_event(_event("content", content="")) is None
        assert to_wire_event(_event("thinking", content=None)) is None

    def test_non_json_safe_result_is_coerced_and_bounded(self):
        class Weird:
            def __str__(self) -> str:
                return "weird-object"

        wire = to_wire_event(_event("tool_result", tool_name="t", result=Weird()))
        assert wire["result"] == "weird-object"

        big = to_wire_event(
            _event("tool_result", tool_name="t", result="x" * (MAX_RESULT_CHARS + 5))
        )
        assert len(big["result"]) < MAX_RESULT_CHARS + 64
        assert "truncated" in big["result"]
        json.dumps(big)  # whole document must be JSON-serializable

    def test_tool_result_payload_dict_never_leaks_internal_keys(self):
        """The client surface flattens the internal tool-pipeline payload onto
        ``result``; only contract fields may cross the wire (UX P3)."""
        payload = {
            "result": "preview",
            "original_result": "full output text",
            "success": True,
            "elapsed": 0.12,
            "was_pruned": False,
            "follow_up_suggestions": [{"cmd": "victor x"}],
            "arguments": {"path": "a.py"},
            "tool_call_id": "call-7",
        }
        wire = to_wire_event(_event("tool_result", tool_name="read", result=payload))

        assert wire["result"] == "full output text"
        assert wire["success"] is True
        assert wire["elapsed_ms"] == 120
        assert wire["call_id"] == "call-7"
        assert set(wire) <= {
            "v",
            "event",
            "tool",
            "success",
            "result",
            "call_id",
            "elapsed_ms",
            "truncated",
        }
        assert "follow_up_suggestions" not in json.dumps(wire)
        assert "original_result" not in json.dumps(wire)

    def test_tool_result_payload_pruned_flags_truncated(self):
        payload = {"result": "short", "original_result": None, "success": True, "was_pruned": True}
        wire = to_wire_event(_event("tool_result", tool_name="read", result=payload))
        assert wire["truncated"] is True
        assert wire["result"] == "short"

    def test_tool_result_payload_failure_wins(self):
        payload = {"result": "boom", "success": False}
        wire = to_wire_event(_event("tool_result", tool_name="shell", result=payload))
        assert wire["success"] is False

    def test_tool_call_carries_call_id_when_present(self):
        event = _event("tool_call", tool_name="read", arguments={"path": "a.py"})
        event.metadata = {"tool_call_id": "call-1"}
        wire = to_wire_event(event)
        assert wire["call_id"] == "call-1"

        # absent id -> key omitted (additive-optional, not null-filled)
        bare = to_wire_event(_event("tool_call", tool_name="read"))
        assert "call_id" not in bare

    def test_scalar_tool_result_keeps_legacy_shape(self):
        wire = to_wire_event(_event("tool_result", tool_name="read", result="ok"))
        assert wire["result"] == "ok"
        assert "elapsed_ms" not in wire
        assert "truncated" not in wire

    def test_wire_truncation_sets_truncated_flag(self):
        big = to_wire_event(
            _event("tool_result", tool_name="t", result="x" * (MAX_RESULT_CHARS + 5))
        )
        assert big["truncated"] is True

    def test_encode_sse_frames_one_record(self):
        frame = encode_sse({"v": 1, "event": "content", "content": "hi"})
        assert frame.startswith("data: ")
        assert frame.endswith("\n\n")
        assert json.loads(frame[len("data: ") :].strip())["event"] == "content"


class _FakeClient:
    def __init__(self, events, explode_after=None):
        self._events = events
        self._explode_after = explode_after

    async def stream(self, message):
        for i, event in enumerate(self._events):
            if self._explode_after is not None and i == self._explode_after:
                raise RuntimeError("provider died")
            yield event


async def _collect(client):
    return [w async for w in stream_wire_events(client, "hi")]


class TestStreamWireEvents:
    async def test_full_stream_terminates_with_exactly_one_stream_end(self):
        client = _FakeClient(
            [
                _event("thinking", content="t"),
                _event("tool_call", tool_name="read"),
                _event("tool_result", tool_name="read", result="ok"),
                _event("content", content="answer"),
                _event("stream_end"),
            ]
        )
        wires = await _collect(client)
        assert [w["event"] for w in wires] == [
            "thinking",
            "tool_call",
            "tool_result",
            "content",
            "stream_end",
        ]

    async def test_missing_stream_end_is_appended(self):
        wires = await _collect(_FakeClient([_event("content", content="a")]))
        assert [w["event"] for w in wires] == ["content", "stream_end"]

    async def test_exception_surfaces_as_error_then_stream_end(self):
        client = _FakeClient(
            [_event("content", content="partial"), _event("content", content="never")],
            explode_after=1,
        )
        wires = await _collect(client)
        assert [w["event"] for w in wires] == ["content", "error", "stream_end"]
        assert "provider died" in wires[1]["message"]

    async def test_out_of_contract_events_are_filtered(self):
        client = _FakeClient(
            [
                _event("tool_progress", content="50%"),
                _event("content", content="a"),
                _event("stream_end"),
            ]
        )
        wires = await _collect(client)
        assert [w["event"] for w in wires] == ["content", "stream_end"]
