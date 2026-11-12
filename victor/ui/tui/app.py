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
from contextlib import aclosing
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.markup import escape
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Footer, Header, Input
from textual.worker import Worker

from victor.ui.chat_app.event_mapping import RenderAction, RenderKind, map_event
from victor.ui.tui.approval_modal import make_tui_approval_handler
from victor.ui.tui.conversation import ConversationLog
from victor.ui.tui.diff_pane import DiffPane, extract_edit_diff
from victor.ui.tui.keybindings import load_keybindings
from victor.ui.tui.palette import CommandPalette, HelpScreen
from victor.ui.tui.phase import PhaseTracker
from victor.ui.tui.prompt_input import PromptInput
from victor.ui.tui.prompt_queue import PromptQueue
from victor.ui.tui.session_bar import TokenUsage
from victor.ui.tui.status_bar import StatusBar
from victor.ui.tui.themes import DEFAULT_THEME, next_theme, register_all, resolve_theme
from victor.ui.tui.token_usage_bar import TokenUsageBar
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

    # Enable mouse support for scrollbar interaction
    ENABLE_COMMAND_PALETTE = True
    MOUSE_ENABLED = True

    class QueueChanged(Message):
        """A prompt was enqueued, consumed, or cleared; re-project the status."""

    class TurnFinished(Message):
        """The turn identified by ``serial`` completed (success, error, interrupt)."""

        def __init__(self, serial: int) -> None:
            super().__init__()
            self.serial = serial

    # Resolve at import: if the stylesheet is missing (e.g. an incomplete wheel
    # that dropped ui/tui/*.tcss), start unstyled instead of crashing Textual
    # with StylesheetError. Packaging guards keep the file present in real dists.
    CSS_PATH = "styles.tcss" if (Path(__file__).with_name("styles.tcss")).exists() else None
    TITLE = "victor"
    # Read once at import; user overrides come from ~/.victor/keybindings.json.
    BINDINGS = [
        Binding(key, action, description) for key, action, description in load_keybindings()
    ]

    def __init__(
        self,
        *,
        client: Any,
        settings: Any,
        agent: Any = None,
        mode: Optional[str] = None,
        tool_budget: Optional[int] = None,
        theme: str = DEFAULT_THEME,
        bootstrap: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self._client = client
        self._agent = agent
        self._settings = settings
        self._mode = mode or "—"
        self._tool_budget = tool_budget
        self._initial_theme = resolve_theme(theme)

        # Shell-first startup: when ``bootstrap`` is given, the app mounts
        # immediately and finishes session initialization (agent creation,
        # resume, turn limits) in a worker; the prompt stays disabled until it
        # resolves. Without it, the session is already initialized (REPL-style
        # callers) and the app is usable on mount.
        self._bootstrap = bootstrap
        self._session_ready = bootstrap is None
        self._bootstrap_error: Optional[str] = None
        self._bootstrap_result: Any = None

        self._phase = PhaseTracker()
        self._turn_worker: Optional[Worker[None]] = None
        self._turn_serial = 0

        # Prompts typed while a turn runs; drained FIFO when the run completes.
        self._queue = PromptQueue(on_change=self._on_queue_change)

        # Watchdog / "waiting on model" state.
        self._wait_timer: Any = None
        self._wait_start: Optional[float] = None
        self._waiting_seconds: Optional[int] = None

        # Last-turn cost figures for the footer.
        self._last_tokens: Optional[int] = None
        self._last_cost: Optional[float] = None
        # Canonical last-turn report (full cache/hit-rate fidelity for the bar)
        # and the session cost accumulator — updated once per turn in
        # _apply_last_turn_cost, never per stream event.
        self._last_report: Optional[dict] = None
        self._session_cost_usd: float = 0.0

        # Detailed token usage tracking.
        self._token_usage = TokenUsage()
        self._context_pct: Optional[float] = None
        self._context_used_tokens: Optional[int] = None
        self._context_limit_tokens: Optional[int] = None

        # A capturing console so shared slash commands can render into the log.
        self._capture_buffer = io.StringIO()
        self._capture_console = Console(file=self._capture_buffer, force_terminal=False, width=100)
        self._slash: Any = None
        # Serializes slash dispatches: each runs against its own capture
        # console (execute_captured), so concurrent commands can neither
        # interleave output nor truncate a shared buffer mid-render.
        self._slash_lock = asyncio.Lock()

    # ── layout ────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Vertical(id="body"):
            with Horizontal(id="main-area"):
                yield ConversationLog(id="conversation")
                yield DiffPane(id="diff-pane")
            yield StatusBar(id="status-bar")
            yield TokenUsageBar(id="token-usage-bar")
            yield PromptInput(placeholder="Message…  (/help for commands)", id="prompt")
        yield Footer()

    async def on_mount(self) -> None:
        # Register + apply the TUI themes (styles.tcss references their variables).
        register_all(self)
        self.theme = self._initial_theme
        self.sub_title = self._model_label()
        # Terminal-native HITL: register before the first turn — and before the
        # bootstrap worker creates the agent (client.py requires the approval
        # handler to exist before Agent.create()).
        try:
            self._client.set_approval_handler(make_tui_approval_handler(self))
        except Exception:  # noqa: BLE001 - approval is best-effort, never fatal
            pass
        if self._bootstrap is not None:
            # Shell-first: the UI is up; finish session init in the background
            # with visible progress instead of blocking before the app mounts.
            prompt = self.query_one("#prompt", Input)
            prompt.disabled = True
            self.query_one("#conversation", ConversationLog).write("[dim]initializing session…[/]")
            self._update_status()
            self.run_worker(self._run_bootstrap(), group="bootstrap", exclusive=True)
        # Shared slash commands, captured into the conversation log.
        try:
            from victor.ui.slash.handler import SlashCommandHandler

            self._slash = SlashCommandHandler(self._capture_console, self._settings, self._agent)
        except Exception:  # noqa: BLE001
            self._slash = None
        await self._refresh_sidebar()
        self._update_status()
        self.query_one("#diff-pane", DiffPane).display = False
        if self._session_ready:
            self.query_one("#prompt", Input).focus()

    async def _run_bootstrap(self) -> None:
        """Finish session initialization after the UI is already visible."""
        started = time.monotonic()
        bootstrap = self._bootstrap
        assert bootstrap is not None  # guarded by _session_ready at mount
        try:
            result = await bootstrap()
        except Exception as exc:  # noqa: BLE001 - surfaced, then a clean exit
            self._bootstrap_error = str(exc) or exc.__class__.__name__
            self.query_one("#conversation", ConversationLog).write(
                f"[red]⚠ initialization failed: {escape(self._bootstrap_error)}[/]"
            )
            self.call_after_refresh(self.exit)
            return
        self._bootstrap_result = result
        if result is not None:
            self._agent = result
            # The handler was built during on_mount against the then-None
            # agent (shell-first bootstrap); rebind it or every agent-bound
            # slash command reports "requires an active session" forever.
            if self._slash is not None:
                self._slash.set_agent(result)
        self._session_ready = True
        self.query_one("#conversation", ConversationLog).write(
            f"[green]✓ ready in {time.monotonic() - started:.1f}s[/]"
        )
        await self._refresh_sidebar()
        self.sub_title = self._model_label()
        self._update_status()
        prompt = self.query_one("#prompt", Input)
        prompt.disabled = False
        prompt.focus()

    # ── input handling ────────────────────────────────────────────

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "prompt" or not self._session_ready:
            return
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        if text.startswith("/"):
            self.run_worker(self._dispatch_slash(text), group="slash")
            return
        # The prompt stays live during a turn: a submission mid-run queues
        # instead of starting a second turn, and runs when the queue drains.
        if self._turn_running():
            self._queue.enqueue(text)
            return
        self._start_turn(text)

    def _turn_running(self) -> bool:
        """True while a turn worker is live (race-free via ``Worker.is_finished``)."""
        worker = self._turn_worker
        return worker is not None and not worker.is_finished

    def _start_turn(self, text: str, *, queued: bool = False) -> None:
        self._turn_serial += 1
        serial = self._turn_serial
        self._turn_worker = self.run_worker(
            self._run_turn(text, queued=queued, serial=serial),
            exclusive=True,
            group="turn",
        )

    def _on_queue_change(self, _queue: PromptQueue) -> None:
        self.post_message(self.QueueChanged())

    # Handler names follow Textual's nested-message rule: camel_to_snake of the
    # enclosing class ("VictorTUIApp" → "victor_tuiapp", no break before the
    # acronym) + the message name.
    def on_victor_tuiapp_queue_changed(self, event: QueueChanged) -> None:
        self._update_status()

    def on_victor_tuiapp_turn_finished(self, event: TurnFinished) -> None:
        """Drain the prompt queue: one queued prompt per finished turn.

        Ownership is decided by turn serial, not ``Worker.is_finished`` — the
        message is posted from the finishing worker's ``finally``, before the
        worker's state transitions. A serial mismatch means a newer turn
        already owns the "turn" group. Starting the next worker from the
        message loop (not from the ``finally``) also means the
        ``exclusive=True`` start can never cancel a worker that is still
        closing.
        """
        if event.serial != self._turn_serial:
            return  # a newer turn already owns the "turn" group
        next_text = self._queue.consume()
        if next_text is not None:
            self._start_turn(next_text, queued=True)

    # ── the turn loop ─────────────────────────────────────────────

    async def _run_turn(self, message: str, *, queued: bool = False, serial: int = 0) -> None:
        convo = self.query_one("#conversation", ConversationLog)
        if queued:
            convo.write("[dim]▲ running queued prompt[/]")
        convo.begin_turn(message)
        self.query_one("#diff-pane", DiffPane).clear_edits()
        self._phase.begin_turn()
        self._update_status()
        try:
            # aclosing parity with the Chainlit surface: on interrupt the
            # generator chain must close deterministically, not at GC time.
            async with aclosing(
                with_stall_watchdog(
                    self._stream_actions(message),
                    _WATCHDOG_TIMEOUT,
                    self._on_stall,
                    self._on_resume,
                )
            ) as guarded:
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
            await self._refresh_sidebar()
            self._phase.reset()
            self._update_status()
            self.post_message(self.TurnFinished(serial))

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
        parts = text[1:].split()
        name = parts[0].lower() if parts else ""
        if name in ("help", "?", "commands"):
            self.push_screen(HelpScreen())
            return
        if name in ("quit", "exit", "q"):
            await self.action_quit()
            return
        if name in ("clear", "cls"):
            self.query_one("#conversation", ConversationLog).clear()
            return
        if name == "queue":
            self._show_queue(text)
            return
        await self._run_captured_slash(text)

    def _show_queue(self, text: str) -> None:
        """List pending queued prompts, or clear them with ``/queue clear``."""
        convo = self.query_one("#conversation", ConversationLog)
        args = text[1:].split()[1:] if len(text) > 1 else []
        if args and args[0].lower() == "clear":
            dropped = self._queue.clear()
            if dropped:
                convo.write(
                    f"[dim]queue cleared ({dropped} prompt{'s' if dropped != 1 else ''})[/]"
                )
            else:
                convo.write("[dim]queue is already empty[/]")
            return
        items = self._queue.snapshot()
        if not items:
            convo.write("[dim]queue is empty[/]")
            return
        convo.write("[bold]queued prompts (oldest first):[/]")
        for i, item in enumerate(items, 1):
            convo.write(f"  {i}. {escape(item)}")

    async def _run_captured_slash(self, text: str) -> None:
        convo = self.query_one("#conversation", ConversationLog)
        if self._slash is None:
            convo.write("[dim]command handler unavailable in this session[/]")
            return
        # One lock, per-command buffers: overlapping dispatches render fully
        # and in dispatch order instead of racing a shared StringIO.
        async with self._slash_lock:
            try:
                output = await self._slash.execute_captured(text)
            except Exception as exc:  # noqa: BLE001
                convo.write(f"[red]⚠ {escape(str(exc))}[/]")
                return
        if output:
            for line in output.splitlines():
                convo.write(escape(line))
        await self._refresh_sidebar()
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
        """Interrupt the running turn; when idle, flush the prompt queue.

        With a turn live, ESC cancels it — the queue is preserved and the next
        queued prompt starts when the cancellation settles. With no turn
        running, ESC is the explicit way to throw the queue away.
        """
        worker = self._turn_worker
        if worker is not None and not worker.is_finished:
            worker.cancel()
            return
        dropped = self._queue.clear()
        if dropped:
            self.notify(
                f"Queue cleared ({dropped} prompt{'s' if dropped != 1 else ''}).",
                timeout=3,
            )

    def action_command_palette(self) -> None:
        self.push_screen(CommandPalette(), self._on_palette_pick)

    def action_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_toggle_diff(self) -> None:
        """Toggle diff pane expand/collapse, or reveal if hidden."""
        pane = self.query_one("#diff-pane", DiffPane)
        if pane.has_edits:
            if not pane.display:
                pane.display = True
            else:
                pane.toggle_expand()

    def action_diff_next(self) -> None:
        self.query_one("#diff-pane", DiffPane).cycle()

    def action_cycle_theme(self) -> None:
        self.theme = next_theme(self.theme)

    def action_clear(self) -> None:
        self.query_one("#conversation", ConversationLog).clear()

    def action_copy(self) -> None:
        """Copy the mouse selection, or — with nothing selected — the last reply.

        Highlight any text with the mouse to copy exactly that span across the
        panes; with no active selection, ``ctrl+c`` copies the whole of the last
        assistant response. Writes to the terminal clipboard via OSC 52, so it
        works over SSH and inside tmux.
        """
        label = "selection"
        try:
            text = self.screen.get_selected_text() or ""
        except Exception:  # noqa: BLE001 - selection is best-effort, never fatal
            text = ""
        if not text:
            label = "response"
            try:
                text = self.query_one("#conversation", ConversationLog).last_response_text()
            except Exception:  # noqa: BLE001
                text = ""
        if not text:
            self.notify("Nothing to copy yet.", severity="warning", timeout=3)
            return
        self.copy_to_clipboard(text)
        self.notify(f"Copied {label} to clipboard ({len(text)} chars).", timeout=3)

    def _on_palette_pick(self, command: Optional[str]) -> None:
        if command:
            self.run_worker(self._dispatch_slash(command), group="slash")

    # ── status + sidebar projection ───────────────────────────────

    def _apply_last_turn_cost(self) -> None:
        """Extract detailed token usage and cost from the last turn."""
        try:
            report = self._client.get_last_turn_cost()
        except Exception:  # noqa: BLE001
            return
        if not isinstance(report, dict) or not report:
            return

        # Extract total tokens
        tokens = report.get("total_tokens")
        if tokens is None:
            tokens = (report.get("api_prompt_tokens") or report.get("prompt_tokens") or 0) + (
                report.get("api_completion_tokens") or report.get("completion_tokens") or 0
            )
        self._last_tokens = int(tokens) if tokens else None

        # Extract cost. Accumulation lives HERE (once per turn) — never in the
        # usage bar, which refreshes on every stream event.
        cost = report.get("total_cost_usd")
        self._last_cost = float(cost) if isinstance(cost, (int, float)) else None
        if self._last_cost is not None:
            self._session_cost_usd += self._last_cost

        # Keep the canonical report so the usage bar can render full fidelity
        # (cache read/write, hit rate, request count, reasoning split).
        self._last_report = report

        # Extract detailed token usage breakdown
        self._token_usage = TokenUsage(
            prompt_tokens=int(report.get("api_prompt_tokens") or report.get("prompt_tokens") or 0),
            completion_tokens=int(
                report.get("api_completion_tokens") or report.get("completion_tokens") or 0
            ),
            reasoning_tokens=int(report.get("reasoning_tokens") or 0),
            cached_tokens=int(report.get("cache_read_tokens") or report.get("cached_tokens") or 0),
            total_tokens=int(tokens) if tokens else 0,
        )

    def _update_status(self) -> None:
        # Get session/model info for status bar
        session_id = _short(getattr(self._agent, "active_session_id", None))
        model = self._safe(lambda: self._client.model)
        provider = self._safe(lambda: self._client.provider_name)
        profile = "—"
        cwd = None
        try:
            if hasattr(self._settings, "profile") and self._settings.profile:
                profile = self._settings.profile
        except Exception:
            pass

        # Get current working directory
        try:
            if hasattr(self._settings, "cwd") and self._settings.cwd:
                cwd = str(self._settings.cwd)
            elif hasattr(self._agent, "_orchestrator") and self._agent._orchestrator:
                orch = self._agent._orchestrator
                if hasattr(orch, "settings") and hasattr(orch.settings, "cwd"):
                    cwd = str(orch.settings.cwd)
        except Exception:
            pass

        self.query_one("#status-bar", StatusBar).set_status(
            phase_label=self._phase.label(),
            tool_count=self._phase.tool_count,
            waiting_seconds=self._waiting_seconds,
            queued=len(self._queue),
            session_id=session_id,
            model=model,
            provider=provider,
            profile=profile,
            cwd=cwd,
        )
        self._update_token_usage_bar()

    def _update_token_usage_bar(self) -> None:
        """Refresh the token usage bar above the prompt input."""
        try:
            bar = self.query_one("#token-usage-bar", TokenUsageBar)
        except Exception:  # noqa: BLE001
            return
        report = self._last_report or {}
        context_used: Optional[int] = None
        context_limit: Optional[int] = None
        try:
            window = self._client.context_window()
            if window and window > 0:
                context_limit = int(window)
                if isinstance(self._context_limit_tokens, int) and self._context_limit_tokens > 0:
                    context_limit = self._context_limit_tokens
                if isinstance(self._context_used_tokens, int) and self._context_used_tokens >= 0:
                    context_used = self._context_used_tokens
        except Exception:  # noqa: BLE001
            context_limit = None
        bar.set_usage(
            context_used_tokens=context_used,
            context_limit_tokens=context_limit,
            context_used_pct=self._context_pct,
            prompt_tokens=report.get("api_prompt_tokens") or report.get("prompt_tokens"),
            completion_tokens=report.get("api_completion_tokens")
            or report.get("completion_tokens"),
            reasoning_tokens=report.get("reasoning_tokens")
            or (self._token_usage.reasoning_tokens or None),
            cached_tokens=report.get("cache_read_tokens")
            or (self._token_usage.cached_tokens or None),
            cache_write_tokens=report.get("cache_write_tokens") or None,
            cache_hit_rate=report.get("cache_hit_rate"),
            total_tokens=self._last_tokens,
            cost_usd=self._last_cost,
            # Authoritative session figure wins; fall back to the once-per-turn
            # accumulator so pre-session_cost_usd transports still show a total.
            session_cost_usd=(
                report.get("session_cost_usd") or (round(self._session_cost_usd, 6) or None)
            ),
            model=self._model_label(),
        )

    async def _refresh_sidebar(self) -> None:
        """Update context telemetry for the token usage bar."""
        context_used, context_total, context_pct = await self._compute_context_usage()
        self._context_pct = context_pct
        # Cache absolute token counts for the usage bar's gauge.
        if isinstance(context_used, int) and context_used >= 0:
            self._context_used_tokens = context_used
        if isinstance(context_total, int) and context_total > 0:
            self._context_limit_tokens = context_total

    async def _compute_context_usage(self) -> tuple[Optional[int], Optional[int], Optional[float]]:
        """Read current context telemetry through the framework client."""
        try:
            context_window = self._client.context_window()
            if not context_window or context_window <= 0:
                return None, None, None
            if not self._session_ready:
                return None, context_window, None
            metrics = await self._client.get_session_metrics()
            used = metrics.get("context_tokens")
            if isinstance(used, int) and used >= 0:
                return used, context_window, (used / context_window) * 100.0
            return None, context_window, None
        except Exception:  # noqa: BLE001 - optional telemetry must not stop a turn
            return None, None, None

    def _model_label(self) -> str:
        return self._safe(lambda: self._client.model) or "victor"

    @staticmethod
    def _safe(getter: Any) -> Optional[str]:
        try:
            value = getter()
            return str(value) if value else None
        except Exception:  # noqa: BLE001
            return None
