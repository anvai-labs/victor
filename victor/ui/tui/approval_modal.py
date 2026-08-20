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

"""Terminal-native HITL approval for the TUI (ADR-021).

Policy ``ASK`` verdicts are delivered out-of-band via
``VictorClient.set_approval_handler`` — a blocking async callback, not a stream
event. :func:`make_tui_approval_handler` returns such a callback that pushes an
:class:`ApprovalScreen` and awaits the user's decision through an
``asyncio.Future``, so CLI users approve/reject in the terminal instead of a
browser. A per-request timer enforces ``timeout_seconds`` (fail-safe reject).
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional, Tuple

from rich.markup import escape
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

from victor.framework.hitl import ApprovalRequest, ApprovalStatus
from victor.ui.rendering.markdown_presenters import tool_call_summary
from victor.ui.rendering.tool_preview import ToolPreviewRenderer

#: Preview lines shown for the pending operation (expanded with the details key).
_PREVIEW_LINES = 8
_PREVIEW_LINES_EXPANDED = 40

#: What the handler returns to the framework: (status, message, responder).
ApprovalOutcome = Tuple[ApprovalStatus, Optional[str], Optional[str]]

_RESPONDER = "tui_user"


def _extract_tool(context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Pull the tool name and arguments out of an approval request context."""
    tool = context.get("tool") or context.get("tool_name") or "tool"
    args = context.get("arguments")
    if not isinstance(args, dict):
        args = context.get("args") if isinstance(context.get("args"), dict) else {}
    return str(tool), dict(args)


class ApprovalScreen(ModalScreen[ApprovalStatus]):
    """Modal that renders a pending :class:`ApprovalRequest` and captures a decision."""

    BINDINGS = [
        ("a", "approve", "Approve"),
        ("r", "reject", "Reject"),
        ("escape", "reject", "Reject"),
        ("v", "toggle_details", "Details"),
    ]

    def __init__(self, request: ApprovalRequest) -> None:
        super().__init__()
        self._request = request
        self._preview = ToolPreviewRenderer()
        self._expanded = False
        self._decided = False

    def _title_markup(self) -> str:
        """Build the title line. ADR-023: prefix a member tag when a team member asked."""
        member = self._request.context.get("member_id")
        member_tag = f"[cyan]\\[member {escape(str(member))}][/] " if member else ""
        return f"[bold yellow]⚠ Approval required[/]  {member_tag}{escape(self._request.title)}"

    def compose(self) -> ComposeResult:
        with Vertical(id="approval-dialog"):
            yield Static(self._title_markup(), id="approval-title")
            if self._request.description:
                yield Static(escape(self._request.description), id="approval-desc")
            yield Static(self._render_preview(), id="approval-preview")
            yield Static(
                f"[dim]\\[a] approve   \\[r] reject   \\[v] details   "
                f"auto-reject in {self._request.timeout_seconds}s[/]",
                id="approval-actions",
            )

    def on_mount(self) -> None:
        # Fail-safe: reject if the user does not respond in time.
        self.set_timer(float(self._request.timeout_seconds), self._on_timeout)

    def _render_preview(self) -> str:
        tool, args = _extract_tool(self._request.context)
        limit = _PREVIEW_LINES_EXPANDED if self._expanded else _PREVIEW_LINES
        header = f"[cyan]{escape(tool_call_summary(tool, args))}[/]"
        try:
            preview = self._preview.render(tool, args, "", limit)
        except Exception:  # noqa: BLE001 - never let preview break the prompt
            return header
        lines = [header]
        for line in preview.lines[:limit]:
            lines.append(line if preview.contains_rich_markup else f"[dim]{escape(line)}[/]")
        return "\n".join(lines)

    def action_toggle_details(self) -> None:
        self._expanded = not self._expanded
        preview = self.query_one("#approval-preview", Static)
        preview.update(self._render_preview())

    def action_approve(self) -> None:
        self._decide(ApprovalStatus.APPROVED)

    def action_reject(self) -> None:
        self._decide(ApprovalStatus.REJECTED)

    def _on_timeout(self) -> None:
        self._decide(ApprovalStatus.TIMEOUT)

    def _decide(self, status: ApprovalStatus) -> None:
        if self._decided:
            return
        self._decided = True
        self.dismiss(status)


def make_tui_approval_handler(app: Any) -> Any:
    """Return an async approval handler that prompts via an :class:`ApprovalScreen`.

    Register the result with ``client.set_approval_handler(...)``. The handler
    blocks the ASK-gated tool call until the user decides (or the screen times
    out), returning ``(status, message, responder)`` to the framework. A dismissal
    with no result fails safe to ``REJECTED``.
    """

    async def handler(request: ApprovalRequest) -> ApprovalOutcome:
        loop = asyncio.get_running_loop()
        future: "asyncio.Future[Optional[ApprovalStatus]]" = loop.create_future()

        def _resolve(result: Optional[ApprovalStatus]) -> None:
            if not future.done():
                future.set_result(result)

        app.push_screen(ApprovalScreen(request), _resolve)
        status = await future
        return (status or ApprovalStatus.REJECTED), None, _RESPONDER

    return handler
