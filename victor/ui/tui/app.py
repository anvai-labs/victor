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

"""``VictorTUIApp`` — the interactive multi-pane terminal UI (ADR-020/021).

Composes an agent-state sidebar, a live conversation log, a status footer, and a
prompt. It owns a session (via ``VictorClient``), drives the submit→stream loop,
registers a terminal-native HITL approval handler, and wraps the event stream in
a stall watchdog. It imports only the client/framework public surface and other
``victor.ui`` helpers — never ``victor.agent.*`` (UI-layer boundary).
"""

from __future__ import annotations

import asyncio
import io
import os
import time
from typing import Any, Optional

from rich.console import Console
from rich.markup import escape
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Input
from textual.worker import Worker

from victor.ui.chat_app.event_mapping import RenderAction, RenderKind, map_event
from victor.ui.tui.approval_modal import make_tui_approval_handler
from victor.ui.tui.conversation import ConversationLog
from victor.ui.tui.diff_pane import DiffPane, extract_edit_diff
from victor.ui.tui.keybindings import load_keybindings
from victor.ui.tui.palette import CommandPalette, HelpScreen
from victor.ui.tui.phase import PhaseTracker
from victor.ui.tui.sidebar import AgentState, AgentStatePanel
from victor.ui.tui.status_bar import StatusBar
from victor.ui.tui.watchdog import with_stall_watchdog

#: Seconds of model silence before the status bar shows a "waiting…" indicator.
_WATCHDOG_TIMEOUT = 12.0


def _short(value: Optional[str]) -> str:
    """Abbreviate a session id for the sidebar."""
    if not value:
        return "—"
    return value[:8]


def _abbrev_path(path: str, home: Optional[str] = None) -> str:
    """Render a path with ``~`` for the home directory."""
    home = home if home is not None else os.path.expanduser("~")
    if home and path.startswith(home):
        return "~" + path[len(home) :]
    return path


class VictorTUIApp(App[None]):
    """Interactive IDE-style TUI: sidebar · conversation · status · prompt."""

    CSS_PATH = "styles.tcss"
    TITLE = "victor"
    # Read once at import; user overrides come from ~/.victor/keybindings.json.
    BINDINGS = [
        Binding(key, action, description) for key, action, description in load_keybindings()
    ]

    def __init__(
        self,
        *,
        client: Any,
        agent: Any,
        settings: Any,
        mode: Optional[str] = None,
        tool_budget: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._client = client
        self._agent = agent
        self._settings = settings
        self._mode = mode or "—"
        self._tool_budget = tool_budget

        self._phase = PhaseTracker()
        self._turn_worker: Optional[Worker[None]] = None

        # Watchdog / "waiting on model" state.
        self._wait_timer: Any = None
        self._wait_start: Optional[float] = None
        self._waiting_seconds: Optional[int] = None

        # Last-turn cost figures for the footer.
        self._last_tokens: Optional[int] = None
        self._last_cost: Optional[float] = None

        # A capturing console so shared slash commands can render into the log.
        self._capture_buffer = io.StringIO()
        self._capture_console = Console(file=self._capture_buffer, force_terminal=False, width=100)
        self._slash: Any = None

    # ── layout ────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="body"):
            yield AgentStatePanel(id="agent-sidebar")
            with Vertical(id="main"):
                yield ConversationLog(id="conversation")
                yield DiffPane(id="diff-pane")
                yield StatusBar(id="status-bar")
                yield Input(placeholder="Message…  (/help for commands)", id="prompt")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = self._model_label()
        # Terminal-native HITL: register before the first turn (late-registration safe).
        try:
            self._client.set_approval_handler(make_tui_approval_handler(self))
        except Exception:  # noqa: BLE001 - approval is best-effort, never fatal
            pass
        # Shared slash commands, captured into the conversation log.
        try:
            from victor.ui.slash.handler import SlashCommandHandler

            self._slash = SlashCommandHandler(self._capture_console, self._settings, self._agent)
        except Exception:  # noqa: BLE001
            self._slash = None
        self._refresh_sidebar()
        self._update_status()
        self.query_one("#diff-pane", DiffPane).display = False
        self.query_one("#prompt", Input).focus()

    # ── input handling ────────────────────────────────────────────

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "prompt":
            return
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        if text.startswith("/"):
            self.run_worker(self._dispatch_slash(text), group="slash")
            return
        # Input is disabled while a turn runs, so this cannot double-fire.
        self._turn_worker = self.run_worker(self._run_turn(text), exclusive=True, group="turn")

    # ── the turn loop ─────────────────────────────────────────────

    async def _run_turn(self, message: str) -> None:
        prompt = self.query_one("#prompt", Input)
        convo = self.query_one("#conversation", ConversationLog)
        prompt.disabled = True
        convo.begin_turn(message)
        self.query_one("#diff-pane", DiffPane).clear_edits()
        self._phase.begin_turn()
        self._update_status()
        try:
            guarded = with_stall_watchdog(
                self._stream_actions(message),
                _WATCHDOG_TIMEOUT,
                self._on_stall,
                self._on_resume,
            )
            async for action in guarded:
                self._phase.update(action)
                convo.feed_action(action)
                self._maybe_capture_diff(action)
                self._update_status()
        except asyncio.CancelledError:
            convo.write("[yellow]⏹ interrupted[/]")
            raise
        except Exception as exc:  # noqa: BLE001 - surface, don't crash the app
            convo.write(f"[red]⚠ {escape(str(exc))}[/]")
        finally:
            self._stop_wait_timer()
            convo.finish_turn()
            self._phase.end_turn()
            self._apply_last_turn_cost()
            self._refresh_sidebar()
            self._phase.reset()
            self._update_status()
            prompt.disabled = False
            prompt.focus()

    async def _stream_actions(self, message: str) -> Any:
        """Yield mapped ``RenderAction``s from the live client stream."""
        async for event in self._client.stream(message):
            yield map_event(event)

    def _maybe_capture_diff(self, action: RenderAction) -> None:
        """Reveal a file edit's diff in the diff pane, if this action is one."""
        if action.kind is not RenderKind.TOOL_END or not action.tool_name:
            return
        arguments = (action.metadata or {}).get("arguments", {})
        edit = extract_edit_diff(action.tool_name, arguments, action.text or "")
        if edit is not None:
            self.query_one("#diff-pane", DiffPane).add_edit(edit)

    # ── watchdog callbacks ────────────────────────────────────────

    def _on_stall(self) -> None:
        self._wait_start = time.monotonic()
        self._waiting_seconds = int(_WATCHDOG_TIMEOUT)
        self._update_status()
        self._wait_timer = self.set_interval(1.0, self._tick_wait)

    def _tick_wait(self) -> None:
        if self._wait_start is not None:
            elapsed = time.monotonic() - self._wait_start
            self._waiting_seconds = int(_WATCHDOG_TIMEOUT + elapsed)
            self._update_status()

    def _on_resume(self) -> None:
        self._stop_wait_timer()
        self._update_status()

    def _stop_wait_timer(self) -> None:
        if self._wait_timer is not None:
            self._wait_timer.stop()
            self._wait_timer = None
        self._wait_start = None
        self._waiting_seconds = None

    # ── slash commands ────────────────────────────────────────────

    async def _dispatch_slash(self, text: str) -> None:
        name = text[1:].split()[0].lower() if len(text) > 1 else ""
        if name in ("help", "?", "commands"):
            self.push_screen(HelpScreen())
            return
        if name in ("quit", "exit", "q"):
            await self.action_quit()
            return
        if name in ("clear", "cls"):
            self.query_one("#conversation", ConversationLog).clear()
            return
        await self._run_captured_slash(text)

    async def _run_captured_slash(self, text: str) -> None:
        convo = self.query_one("#conversation", ConversationLog)
        if self._slash is None:
            convo.write("[dim]command handler unavailable in this session[/]")
            return
        self._capture_buffer.seek(0)
        self._capture_buffer.truncate(0)
        try:
            await self._slash.execute(text)
        except Exception as exc:  # noqa: BLE001
            convo.write(f"[red]⚠ {escape(str(exc))}[/]")
            return
        output = self._capture_buffer.getvalue().strip()
        if output:
            for line in output.splitlines():
                convo.write(escape(line))
        self._refresh_sidebar()
        self.sub_title = self._model_label()

    # ── actions (bound keys) ──────────────────────────────────────

    async def action_quit(self) -> None:
        try:
            from victor.ui.commands.utils import graceful_shutdown

            await graceful_shutdown(self._agent)
        except Exception:  # noqa: BLE001 - shutdown is best-effort on exit
            pass
        self.exit()

    def action_interrupt(self) -> None:
        worker = self._turn_worker
        if worker is not None:
            worker.cancel()

    def action_command_palette(self) -> None:
        self.push_screen(CommandPalette(), self._on_palette_pick)

    def action_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_toggle_sidebar(self) -> None:
        sidebar = self.query_one("#agent-sidebar", AgentStatePanel)
        sidebar.display = not sidebar.display

    def action_toggle_diff(self) -> None:
        pane = self.query_one("#diff-pane", DiffPane)
        if pane.has_edits:
            pane.display = not pane.display

    def action_diff_next(self) -> None:
        self.query_one("#diff-pane", DiffPane).cycle()

    def action_clear(self) -> None:
        self.query_one("#conversation", ConversationLog).clear()

    def _on_palette_pick(self, command: Optional[str]) -> None:
        if command:
            self.run_worker(self._dispatch_slash(command), group="slash")

    # ── status + sidebar projection ───────────────────────────────

    def _apply_last_turn_cost(self) -> None:
        try:
            report = self._client.get_last_turn_cost()
        except Exception:  # noqa: BLE001
            return
        if not isinstance(report, dict) or not report:
            return
        tokens = report.get("total_tokens")
        if tokens is None:
            tokens = (report.get("prompt_tokens") or 0) + (report.get("completion_tokens") or 0)
        self._last_tokens = int(tokens) if tokens else None
        cost = report.get("total_cost_usd")
        self._last_cost = float(cost) if isinstance(cost, (int, float)) else None

    def _update_status(self) -> None:
        self.query_one("#status-bar", StatusBar).set_status(
            phase_label=self._phase.label(),
            tool_count=self._phase.tool_count,
            total_tokens=self._last_tokens,
            cost_usd=self._last_cost,
            waiting_seconds=self._waiting_seconds,
        )

    def _refresh_sidebar(self) -> None:
        state = AgentState(
            session_id=_short(getattr(self._agent, "active_session_id", None)),
            mode=self._mode,
            model=self._safe(lambda: self._client.model) or "—",
            provider=self._safe(lambda: self._client.provider_name) or "—",
            cwd=_abbrev_path(os.getcwd()),
            budget_used=0,
            budget_total=self._tool_budget,
        )
        self.query_one("#agent-sidebar", AgentStatePanel).set_state(state)

    def _model_label(self) -> str:
        return self._safe(lambda: self._client.model) or "victor"

    @staticmethod
    def _safe(getter: Any) -> Optional[str]:
        try:
            value = getter()
            return str(value) if value else None
        except Exception:  # noqa: BLE001
            return None
