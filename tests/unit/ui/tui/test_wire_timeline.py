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

"""UX P5: the TUI wire-event timeline renders the contract with chat semantics.

The golden stream used here mirrors the renderer replay contract test
(``tests/unit/ui/chat_app/test_wire_render_parity.py``): the same recorded
contract stream must render equivalently across surfaces — the TUI timeline is
the third surface (web Chainlit, wire consumers, TUI).
"""

from __future__ import annotations

import io
import json
from types import SimpleNamespace
from typing import Any

from rich.console import Console

from victor.framework.wire_events import to_wire_event
from victor.ui.chat_app.event_mapping import RenderKind, map_wire_event
from victor.ui.tui.wire_timeline import (
    WireTimelineState,
    parse_wire_line,
)


def _to_text(item: Any) -> str:
    """Render a timeline item to plain text (markers are strings; content is a renderable)."""
    if isinstance(item, str):
        return item
    buf = io.StringIO()
    Console(file=buf, width=100, no_color=True).print(item)
    return buf.getvalue()


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


def _golden_wire_stream() -> list[dict]:
    """Same turn shape as the replay parity test, serialized to wire events."""
    ok_payload = {
        "result": "preview",
        "original_result": "full grep output\nline 2",
        "success": True,
        "elapsed": 0.25,
        "tool_call_id": "call-1",
        "arguments": {"pattern": "foo"},
    }
    failed_payload = {
        "result": "exit 1",
        "original_result": "exit 1: no such file",
        "success": False,
        "elapsed": 0.05,
        "tool_call_id": "call-2",
    }
    events = [
        _event("thinking", content="Let me search first."),
        _event("content", content="Searching now. "),
        _event(
            "tool_call",
            tool_name="grep",
            arguments={"pattern": "foo"},
            metadata={"tool_call_id": "call-1"},
        ),
        _event(
            "tool_call",
            tool_name="shell",
            arguments={"cmd": "cat missing"},
            metadata={"tool_call_id": "call-2"},
        ),
        _event("tool_result", tool_name="grep", result=ok_payload, metadata=ok_payload),
        _event("tool_result", tool_name="shell", result=failed_payload, metadata=failed_payload),
        _event("content", content="Found it."),
        _event("error", content="provider hiccup"),
        _event("stream_end"),
    ]
    return [to_wire_event(e) for e in events]


def _render_all(wires: list[dict]) -> list[str]:
    state = WireTimelineState()
    items: list[Any] = []
    for wire in wires:
        items.extend(state.advance(map_wire_event(wire)))
    items.extend(state.flush())
    return [_to_text(i) for i in items]


class TestParseWireLine:
    def test_raw_jsonl_line(self):
        assert parse_wire_line('{"v": 1, "event": "content", "content": "hi"}') == {
            "v": 1,
            "event": "content",
            "content": "hi",
        }

    def test_sse_data_framing(self):
        wire = {"v": 1, "event": "stream_end"}
        assert parse_wire_line(f"data: {json.dumps(wire)}\n") == wire

    def test_noise_returns_none(self):
        assert parse_wire_line("") is None
        assert parse_wire_line("   \n") is None
        assert parse_wire_line("data:") is None
        assert parse_wire_line("not json") is None
        assert parse_wire_line('["a", "list"]') is None


class TestTimelineRendering:
    def test_golden_stream_renders_all_semantics(self):
        text = "\n".join(_render_all(_golden_wire_stream()))

        # reasoning header + content
        assert "🧠 reasoning" in text
        assert "Let me search first." in text
        # tool starts show the summary label, running marker
        assert "🔧" in text and "running…" in text
        # tool ends: success + failure marks, duration labels, result previews
        assert "✓" in text and "✗" in text
        assert "250ms" in text and "50ms" in text
        assert "full grep output" in text
        assert "exit 1: no such file" in text
        # buffered text paragraphs flushed on boundaries
        assert "Searching now." in text
        assert "Found it." in text
        # error surfaced
        assert "⚠ provider hiccup" in text

    def test_text_flushes_before_tool_activity(self):
        lines = _render_all(_golden_wire_stream())
        text_idx = next(i for i, line in enumerate(lines) if "Searching now." in line)
        tool_idx = next(i for i, line in enumerate(lines) if "🔧" in line)
        assert text_idx < tool_idx

    def test_parallel_calls_keep_their_arguments(self):
        # call-2's label at TOOL_END must use call-2's arguments, not call-1's.
        lines = _render_all(_golden_wire_stream())
        failed_line = next(line for line in lines if "✗" in line)
        assert "shell" in failed_line

    def test_truncated_result_notes_pruning(self):
        wire = {
            "v": 1,
            "event": "tool_result",
            "tool": "read",
            "success": True,
            "result": "short",
            "truncated": True,
        }
        lines = WireTimelineState().advance(map_wire_event(wire))
        assert any("truncated" in line for line in lines)

    def test_unknown_and_lifecycle_events_render_nothing(self):
        state = WireTimelineState()
        assert state.advance(map_wire_event({"v": 1, "event": "usage"})) == []
        assert state.advance(map_wire_event({"v": 1, "event": "stream_end"})) == []
        assert state.advance(map_wire_event(None)) == []

    def test_markup_in_content_is_not_interpreted_as_rich_markup(self):
        # Assistant content is rendered as Markdown (the shared renderer), which parses
        # Markdown — not Rich ``[tags]`` — so raw Rich markup in content is never
        # interpreted (no colour injection); the bracketed text survives literally.
        state = WireTimelineState()
        state.advance(map_wire_event({"v": 1, "event": "content", "content": "[red]not markup[/]"}))
        rendered = _to_text(state.flush()[0])
        assert "not markup" in rendered
        assert "[red]" in rendered  # literal brackets, not a colour tag

    def test_member_awaiting_approval_renders_paused_lane(self):
        """ADR-023 pillar 2b: a paused member renders a distinct awaiting-approval lane."""
        wire = {
            "v": 1,
            "event": "member_awaiting_approval",
            "member_id": "m1",
            "content": "run_command",
        }
        lines = WireTimelineState().advance(map_wire_event(wire))
        assert lines
        rendered = lines[0]
        assert "m1" in rendered
        assert "awaiting approval" in rendered
        assert "run_command" in rendered
        assert "⏸" in rendered

    def test_every_render_kind_is_handled(self):
        """Contract: the timeline must consume the full RenderAction vocabulary."""
        from victor.ui.chat_app.event_mapping import RenderAction

        state = WireTimelineState()
        for kind in RenderKind:
            # Must not raise for any kind, known or future.
            state.advance(RenderAction(kind, text="x", tool_name="t"))


class TestWireTimelineWidget:
    def test_feed_wire_writes_into_richlog(self):
        """Drive the Textual widget inside a minimal app context."""
        import asyncio

        from textual.app import App, ComposeResult

        from victor.ui.tui.wire_timeline import WireTimeline

        class _Host(App):
            def compose(self) -> ComposeResult:
                yield WireTimeline(id="wt")

        async def scenario():
            app = _Host()
            async with app.run_test():
                timeline = app.query_one("#wt", WireTimeline)
                for wire in _golden_wire_stream():
                    timeline.feed_wire(wire)
                assert len(timeline.lines) > 0
                # stream_end flushes and draws the terminator
                timeline.feed_line('data: {"v": 1, "event": "stream_end"}')
                assert len(timeline.lines) > 0

        asyncio.run(scenario())
