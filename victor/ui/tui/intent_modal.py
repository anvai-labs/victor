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

"""Task intent confirmation modal for the TUI.

When the agent needs clarification on task intent, this modal presents
recommended options with rationale, allowing the user to confirm, modify,
or discard the current approach. Similar to Claude Code confirmation dialogs.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional

from rich.markup import escape
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static


class IntentAction(Enum):
    """User's decision on the task intent."""

    CONFIRM = "confirm"
    MODIFY = "modify"
    DISCARD = "discard"
    CHAT = "chat"


@dataclass
class IntentOption:
    """A recommended option with rationale."""

    label: str
    rationale: str
    is_recommended: bool = False


@dataclass
class IntentRequest:
    """Request for task intent clarification."""

    task_description: str
    options: List[IntentOption]
    context: Optional[str] = None
    timeout_seconds: int = 60


class IntentConfirmationScreen(ModalScreen[IntentAction]):
    """Modal that presents task intent options and captures user decision."""

    BINDINGS = [
        ("1", "select_first", "First option"),
        ("2", "select_second", "Second option"),
        ("3", "select_third", "Third option"),
        ("d", "discard", "Discard"),
        ("c", "chat", "Chat about it"),
        ("escape", "discard", "Discard"),
    ]

    def __init__(self, request: IntentRequest) -> None:
        super().__init__()
        self._request = request
        self._decided = False
        self._selected_option: Optional[int] = None

    def compose(self) -> ComposeResult:
        with Vertical(id="intent-dialog"):
            # Title
            yield Static(
                "[bold yellow]⚠ Task intent clarification needed[/]",
                id="intent-title",
            )

            # Task description
            yield Static(
                f"[dim]Task:[/] {escape(self._request.task_description)}",
                id="intent-task",
            )

            # Context (if provided)
            if self._request.context:
                yield Static(
                    f"[dim]Context:[/] {escape(self._request.context)}",
                    id="intent-context",
                )

            # Options
            yield Static("[bold]Recommended options:[/]", id="intent-options-title")

            for i, option in enumerate(self._request.options, 1):
                if option.is_recommended:
                    # Highlight recommended option
                    option_text = (
                        f"[bold cyan]{i}. {escape(option.label)} (RECOMMENDED)[/]\n"
                        f"   [dim]{escape(option.rationale)}[/]"
                    )
                else:
                    option_text = (
                        f"[white]{i}. {escape(option.label)}[/]\n"
                        f"   [dim]{escape(option.rationale)}[/]"
                    )
                yield Static(option_text, id=f"intent-option-{i}")

            # Actions
            yield Static(
                "[dim]\\[1-3] select option   \\[d] discard   \\[c] chat about it   "
                f"auto-discard in {self._request.timeout_seconds}s[/]",
                id="intent-actions",
            )

    def on_mount(self) -> None:
        # Fail-safe: discard if the user does not respond in time
        self.set_timer(float(self._request.timeout_seconds), self._on_timeout)

    def action_select_first(self) -> None:
        if len(self._request.options) >= 1:
            self._selected_option = 0
            self._decide(IntentAction.CONFIRM)

    def action_select_second(self) -> None:
        if len(self._request.options) >= 2:
            self._selected_option = 1
            self._decide(IntentAction.CONFIRM)

    def action_select_third(self) -> None:
        if len(self._request.options) >= 3:
            self._selected_option = 2
            self._decide(IntentAction.CONFIRM)

    def action_discard(self) -> None:
        self._decide(IntentAction.DISCARD)

    def action_chat(self) -> None:
        self._decide(IntentAction.CHAT)

    def _on_timeout(self) -> None:
        self._decide(IntentAction.DISCARD)

    def _decide(self, action: IntentAction) -> None:
        if self._decided:
            return
        self._decided = True
        self.dismiss(action)


def make_intent_handler(app: Any) -> Any:
    """Return an async intent handler that prompts via IntentConfirmationScreen.

    The handler blocks until the user decides (or times out), returning the
    selected IntentAction. A dismissal with no result fails safe to DISCARD.
    """

    async def handler(request: IntentRequest) -> IntentAction:
        loop = asyncio.get_running_loop()
        future: "asyncio.Future[Optional[IntentAction]]" = loop.create_future()

        def _resolve(result: Optional[IntentAction]) -> None:
            if not future.done():
                future.set_result(result)

        app.push_screen(IntentConfirmationScreen(request), _resolve)
        action = await future
        return action or IntentAction.DISCARD

    return handler
