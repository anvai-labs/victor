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

"""Render the v1 wire-event stream in Textual surfaces (UX P5).

The third surface of the one-contract design: web (Chainlit) and this TUI
timeline both render the :class:`RenderAction` vocabulary produced by
``victor/ui/chat_app/event_mapping.py`` — the web from ``map_event`` /
``map_wire_event``, the TUI from ``map_wire_event`` over recorded or live
wire-event JSONL. The line formatter mirrors the Chainlit semantics (tool
summary labels, duration suffix, failure marks, truncation notes) so the same
recorded contract stream renders equivalently everywhere.

Layering: the pure pieces (:func:`parse_wire_line`, :class:`WireTimelineState`)
import nothing from Textual and are unit-testable standalone; only
:class:`WireTimeline` touches Textual. Everything consumes the wire contract or
``VictorClient`` surfaces — never ``victor.agent.*`` (enforced by the UI-layer
boundary guard, which scans this package).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from rich.markup import escape

from victor.ui.chat_app.event_mapping import RenderAction, RenderKind, map_wire_event
from victor.ui.rendering.markdown_presenters import tool_call_summary
from victor.ui.rendering.utils import format_duration

#: Longest tool-result preview shown inline in the timeline.
_RESULT_PREVIEW_CHARS = 300


def parse_wire_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse one recorded line into a wire-event dict, or ``None``.

    Accepts both raw JSONL (one wire event per line) and SSE framing
    (``data: {...}``) — so a ``curl -N`` capture of ``POST /chat/stream``
    replays directly. Blank lines and non-JSON noise return ``None``.
    """
    text = line.strip()
    if text.startswith("data:"):
        text = text[len("data:") :].strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


class WireTimelineState:
    """Pure line-producing state machine over :class:`RenderAction` streams.

    Content tokens are buffered into a paragraph and flushed on segment
    boundaries (tool activity, errors, stream end) — mirroring the Chainlit
    app's natural text→tools→text turn flow on a line-oriented surface.
    ``advance``/``flush`` return items ready for ``RichLog.write`` — Rich-markup strings for
    structural markers (tool/member/error lines) and a shared markdown renderable for the
    assistant content paragraph (see :meth:`flush`).
    """

    def __init__(self) -> None:
        self._text_buffer: List[str] = []
        self._in_reasoning = False
        self._pending_args: Dict[str, Dict[str, Any]] = {}

    def advance(self, action: RenderAction) -> List[Any]:
        if action.kind is RenderKind.TOKEN:
            self._in_reasoning = False
            if action.text:
                self._text_buffer.append(action.text)
            return []

        if action.kind is RenderKind.THINKING:
            lines = []
            if not self._in_reasoning:
                self._in_reasoning = True
                lines.append("[dim italic]🧠 reasoning[/]")
            if action.text:
                lines.append(f"[dim]{escape(action.text.strip())}[/]")
            return lines

        if action.kind is RenderKind.TOOL_START:
            lines = self.flush()
            key = action.call_id or action.tool_name or "tool"
            args = action.metadata.get("arguments", {}) if action.metadata else {}
            self._pending_args[key] = args
            label = tool_call_summary(action.tool_name or "tool", args)
            lines.append(f"[cyan]🔧 {escape(label)}[/] [dim]running…[/]")
            return lines

        if action.kind is RenderKind.TOOL_END:
            lines = self.flush()
            key = action.call_id or action.tool_name or "tool"
            args = self._pending_args.pop(key, {})
            label = tool_call_summary(action.tool_name or "tool", args)
            if action.elapsed:
                label = f"{label} · {format_duration(action.elapsed)}"
            mark = "[green]✓[/]" if action.success else "[red]✗[/]"
            lines.append(f"{mark} [bold]{escape(label)}[/]")
            preview = (action.text or "").strip()
            if preview:
                if len(preview) > _RESULT_PREVIEW_CHARS:
                    preview = preview[:_RESULT_PREVIEW_CHARS] + "…"
                first_line = preview.splitlines()[0]
                lines.append(f"  [dim]{escape(first_line)}[/]")
            if action.was_pruned:
                lines.append("  [dim italic](output truncated for length)[/]")
            return lines

        if action.kind is RenderKind.ERROR:
            lines = self.flush()
            lines.append(f"[red]⚠ {escape(action.text)}[/]")
            return lines

        if action.kind is RenderKind.MEMBER_START:
            lines = self.flush()
            member = action.member_id or "member"
            lines.append(f"[magenta]▸[/] [bold]{escape(member)}[/] [dim]started[/]")
            return lines

        if action.kind is RenderKind.MEMBER_END:
            lines = self.flush()
            member = action.member_id or "member"
            mark = "[green]✓[/]" if action.success else "[red]✗[/]"
            status = "done" if action.success else "failed"
            lines.append(f"{mark} [bold]{escape(member)}[/] [dim]{status}[/]")
            return lines

        if action.kind is RenderKind.MEMBER_AWAITING:
            # ADR-023 pillar 2b: durable pause — the member is waiting on human approval.
            lines = self.flush()
            member = action.member_id or "member"
            detail = (action.text or "").strip()
            suffix = f" [yellow]{escape(detail)}[/]" if detail else ""
            lines.append(
                f"[yellow]⏸[/] [bold]{escape(member)}[/] "
                f"[yellow bold]awaiting approval[/]{suffix}"
            )
            return lines

        return []  # IGNORE: lifecycle / future additive events

    def flush(self) -> List[Any]:
        """Emit the buffered assistant paragraph as a Markdown renderable, if any.

        Reuses the shared markdown renderer (``render_markdown_with_hooks`` — the same one
        the CLI's live renderer uses) so assistant markdown (**bold**, headings, lists,
        fenced code, tables) is applied on the TUI too, from a single source of truth rather
        than a TUI-only fork. ``RichLog.write`` accepts the renderable; ``Markdown`` does not
        interpret Rich ``[tags]``, so raw content stays injection-safe (no escaping needed).
        Falls back to escaped plain text if markdown rendering is unavailable.
        """
        self._in_reasoning = False
        if not self._text_buffer:
            return []
        paragraph = "".join(self._text_buffer).strip()
        self._text_buffer = []
        if not paragraph:
            return []
        try:
            from victor.ui.rendering.markdown import render_markdown_with_hooks

            return [render_markdown_with_hooks(paragraph)]
        except Exception:  # pragma: no cover - never let rendering break the turn
            return [escape(paragraph)]


try:  # Textual is a core dependency, but keep the pure pieces importable alone
    from textual.widgets import RichLog
except Exception:  # pragma: no cover - exercised only in broken installs
    RichLog = None  # type: ignore[assignment,misc]


if RichLog is not None:

    class WireTimeline(RichLog):
        """RichLog that renders v1 wire events with chat-surface semantics."""

        def __init__(self, **kwargs: Any) -> None:
            kwargs.setdefault("wrap", True)
            kwargs.setdefault("markup", True)
            super().__init__(**kwargs)
            self._state = WireTimelineState()

        def feed_wire(self, wire: Dict[str, Any]) -> None:
            """Render one wire-event dict (from JSONL replay or a live tail)."""
            for line in self._state.advance(map_wire_event(wire)):
                self.write(line)
            if isinstance(wire, dict) and wire.get("event") == "stream_end":
                for line in self._state.flush():
                    self.write(line)
                self.write("[dim]── stream end ──[/]")

        def feed_line(self, raw_line: str) -> None:
            """Parse and render one recorded line (JSONL or SSE ``data:``)."""
            wire = parse_wire_line(raw_line)
            if wire is not None:
                self.feed_wire(wire)
