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

"""Live conversation log for the interactive TUI.

Extends :class:`~victor.ui.tui.wire_timeline.WireTimeline` with a live path:
``feed_action`` drives the same tested :class:`WireTimelineState` from *live*
``map_event`` output (rather than replayed wire dicts), so the TUI renders the
one-contract ``RenderAction`` vocabulary identically to the web/replay surfaces.
Tool results are enriched to a multi-line inline preview via the shared
:class:`~victor.ui.rendering.tool_preview.ToolPreviewRenderer` (diffs, file reads,
search hits) — a dedicated side-by-side diff pane is deferred (ADR-020).
"""

from __future__ import annotations

from typing import Any, Dict

from rich.markup import escape

from victor.ui.chat_app.event_mapping import RenderAction, RenderKind
from victor.ui.rendering.markdown_presenters import tool_call_summary
from victor.ui.rendering.tool_preview import ToolPreviewRenderer
from victor.ui.rendering.utils import format_duration
from victor.ui.tui.wire_timeline import WireTimeline, WireTimelineState

#: Largest inline tool-result preview shown in the conversation, in lines.
_PREVIEW_MAX_LINES = 12


class ConversationLog(WireTimeline):
    """A ``RichLog`` that renders a live agent turn from ``RenderAction`` events.

    Feed each mapped stream event to :meth:`feed_action`; call :meth:`begin_turn`
    when the user submits and :meth:`finish_turn` when the stream ends.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._pending_args: Dict[str, Dict[str, Any]] = {}
        self._preview = ToolPreviewRenderer()

    def begin_turn(self, user_message: str) -> None:
        """Start a new turn: reset buffers and echo the user's message."""
        self._state = WireTimelineState()
        self._pending_args.clear()
        text = user_message.strip()
        if text:
            self.write(f"[bold cyan]›[/] {escape(text)}")

    def feed_action(self, action: RenderAction) -> None:
        """Render one live :class:`RenderAction` into the log."""
        kind = action.kind
        if kind is RenderKind.TOOL_START:
            key = action.call_id or action.tool_name or "tool"
            args = (action.metadata or {}).get("arguments", {})
            self._pending_args[key] = args if isinstance(args, dict) else {}
            for line in self._state.advance(action):
                self.write(line)
            return
        if kind is RenderKind.TOOL_END:
            # Flush any buffered assistant text, then render the enriched result.
            for line in self._state.flush():
                self.write(line)
            self._write_tool_result(action)
            return
        for line in self._state.advance(action):
            self.write(line)

    def finish_turn(self) -> None:
        """Flush any trailing text and draw a turn separator."""
        for line in self._state.flush():
            self.write(line)
        self.write("[dim]" + "─" * 8 + "[/]")

    # ── internals ─────────────────────────────────────────────────

    def _write_tool_result(self, action: RenderAction) -> None:
        tool = action.tool_name or "tool"
        key = action.call_id or tool
        args = self._pending_args.pop(key, {})
        label = tool_call_summary(tool, args)
        if action.elapsed:
            label = f"{label} · {format_duration(action.elapsed)}"
        mark = "[green]✓[/]" if action.success else "[red]✗[/]"
        self.write(f"{mark} [bold]{escape(label)}[/]")
        self._write_preview(tool, args, action.text or "")
        if action.was_pruned:
            self.write("  [dim italic](output truncated for length)[/]")

    def _write_preview(self, tool: str, args: Dict[str, Any], raw: str) -> None:
        if not raw.strip():
            return
        try:
            preview = self._preview.render(tool, args, raw, _PREVIEW_MAX_LINES)
        except Exception:  # noqa: BLE001 - a preview must never break the turn
            for line in raw.strip().splitlines()[:_PREVIEW_MAX_LINES]:
                self.write(f"  [dim]{escape(line)}[/]")
            return
        if preview.header:
            self.write(f"  [dim]{escape(preview.header)}[/]")
        shown = preview.lines[:_PREVIEW_MAX_LINES]
        for line in shown:
            if preview.contains_rich_markup:
                self.write(f"  {line}")
            else:
                self.write(f"  [dim]{escape(line)}[/]")
        hidden = preview.total_line_count - len(shown)
        if hidden > 0:
            self.write(f"  [dim italic](+{hidden} more lines)[/]")
