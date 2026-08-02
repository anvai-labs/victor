# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Renderer replay contract (UX P3).

One golden client stream, two renderers: the in-process mapper
(``map_event`` over raw ``VictorClient.stream()`` events) and the wire mapper
(``map_wire_event`` over ``to_wire_event`` serializations of the same events)
must agree on render semantics — same action kinds, same tool identity and
correlation, same success/duration — differing only where the wire contract
intentionally bounds payloads. This is the parity gate P5 (TUI) reuses when it
adopts the wire contract.
"""

from __future__ import annotations

from types import SimpleNamespace

from victor.framework.wire_events import to_wire_event
from victor.ui.chat_app.event_mapping import RenderKind, map_event, map_wire_event


def _event(event_type: str, **kwargs) -> SimpleNamespace:
    defaults = {
        "content": None,
        "tool_name": None,
        "arguments": None,
        "result": None,
        "success": True,
        "metadata": {},
    }
    defaults.update(kwargs)
    return SimpleNamespace(event_type=event_type, **defaults)


def _golden_stream() -> list[SimpleNamespace]:
    """A representative turn: reasoning → text → parallel tools → text → end."""
    tool_payload = {
        "result": "preview (use /expand)",
        "original_result": "full grep output\nline 2",
        "success": True,
        "elapsed": 0.25,
        "was_pruned": False,
        "follow_up_suggestions": [{"cmd": "victor read a.py"}],
        "arguments": {"pattern": "foo"},
        "tool_call_id": "call-1",
    }
    failed_payload = {
        "result": "exit 1",
        "original_result": "exit 1: no such file",
        "success": False,
        "elapsed": 0.05,
        "tool_call_id": "call-2",
    }
    return [
        _event("thinking", content="Let me search first."),
        _event("content", content="Searching now. "),
        _event(
            "tool_call",
            tool_name="grep",
            arguments={"pattern": "foo"},
            metadata={"tool_call_id": "call-1", "arguments": {"pattern": "foo"}},
        ),
        _event(
            "tool_call",
            tool_name="shell",
            arguments={"cmd": "cat missing"},
            metadata={"tool_call_id": "call-2", "arguments": {"cmd": "cat missing"}},
        ),
        _event("tool_result", tool_name="grep", result=tool_payload, metadata=tool_payload),
        _event("tool_result", tool_name="shell", result=failed_payload, metadata=failed_payload),
        _event("content", content="Found it."),
        _event("error", content="provider hiccup"),
        _event("stream_end"),
    ]


def _replay_both(events):
    in_process = [map_event(e) for e in events]
    wire_side = [map_wire_event(to_wire_event(e)) for e in events]
    return in_process, wire_side


class TestReplayParity:
    def test_action_kind_sequences_agree(self):
        in_process, wire_side = _replay_both(_golden_stream())
        assert [a.kind for a in in_process] == [a.kind for a in wire_side]

    def test_kind_sequence_is_the_expected_turn_shape(self):
        in_process, _ = _replay_both(_golden_stream())
        assert [a.kind for a in in_process] == [
            RenderKind.THINKING,
            RenderKind.TOKEN,
            RenderKind.TOOL_START,
            RenderKind.TOOL_START,
            RenderKind.TOOL_END,
            RenderKind.TOOL_END,
            RenderKind.TOKEN,
            RenderKind.ERROR,
            RenderKind.IGNORE,  # stream_end is lifecycle on both paths
        ]

    def test_tool_identity_and_correlation_agree(self):
        in_process, wire_side = _replay_both(_golden_stream())
        for raw, wired in zip(in_process, wire_side):
            if raw.kind in (RenderKind.TOOL_START, RenderKind.TOOL_END):
                assert wired.tool_name == raw.tool_name
                assert wired.call_id == raw.call_id
                assert wired.call_id is not None  # parallel calls stay correlated

    def test_tool_outcome_and_duration_agree(self):
        in_process, wire_side = _replay_both(_golden_stream())
        ends = [
            (raw, wired)
            for raw, wired in zip(in_process, wire_side)
            if raw.kind == RenderKind.TOOL_END
        ]
        assert len(ends) == 2
        for raw, wired in ends:
            assert wired.success == raw.success
            assert abs(wired.elapsed - raw.elapsed) < 0.001

        assert ends[0][1].success is True
        assert ends[1][1].success is False

    def test_text_content_agrees_where_contract_is_unbounded(self):
        in_process, wire_side = _replay_both(_golden_stream())
        for raw, wired in zip(in_process, wire_side):
            if raw.kind in (RenderKind.TOKEN, RenderKind.THINKING, RenderKind.ERROR):
                assert wired.text == raw.text
            if raw.kind == RenderKind.TOOL_END:
                # Both prefer the full human-readable output; the wire may
                # bound it, so equality holds up to truncation.
                assert raw.text.startswith(wired.text[: len(raw.text)]) or wired.text.startswith(
                    raw.text[: len(wired.text)]
                )

    def test_wire_mapper_ignores_future_additive_events(self):
        assert map_wire_event({"v": 1, "event": "usage", "tokens": 5}).kind == RenderKind.IGNORE
        assert map_wire_event({"v": 1, "event": "stream_end"}).kind == RenderKind.IGNORE
        assert map_wire_event(None).kind == RenderKind.IGNORE
        assert map_wire_event("data:").kind == RenderKind.IGNORE
