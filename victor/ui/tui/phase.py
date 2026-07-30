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

"""Inferred agent-loop phase for the TUI status bar.

The live event stream (``VictorClient.stream`` → ``map_event`` →
:class:`~victor.ui.chat_app.event_mapping.RenderAction`) does **not** carry the
runtime's internal PERCEIVE→PLAN→ACT→EVALUATE→DECIDE phase — that state never
crosses the client boundary. Until a framework phase-event enhancement exists
(ADR-021, gated on a FEP), :class:`PhaseTracker` *infers* a coarse, honest phase
from the kinds of events that do arrive, purely for a legible status line.

This module imports nothing from Textual and is unit-testable standalone.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from victor.ui.chat_app.event_mapping import RenderAction, RenderKind


class Phase(str, Enum):
    """Coarse, inferred agent-loop phase (display-only)."""

    IDLE = "idle"  # no turn in flight
    WAITING = "waiting"  # turn started, first event not yet seen
    PLANNING = "planning"  # model is reasoning (THINKING events)
    ACTING = "acting"  # a tool is running (between TOOL_START and TOOL_END)
    RESPONDING = "responding"  # model is emitting the answer (TOKEN events)
    DONE = "done"  # turn finished


class PhaseTracker:
    """Infer a coarse loop phase from the live ``RenderAction`` stream.

    Call :meth:`begin_turn` when a message is submitted, feed every
    :class:`RenderAction` to :meth:`update`, and :meth:`end_turn` when the stream
    completes. :attr:`phase`, :attr:`active_tool`, and :attr:`tool_count` back the
    status bar; :meth:`label` renders a short human string.
    """

    def __init__(self) -> None:
        self._phase: Phase = Phase.IDLE
        self._active_tool: Optional[str] = None
        self._tool_count: int = 0

    @property
    def phase(self) -> Phase:
        """The current inferred phase."""
        return self._phase

    @property
    def active_tool(self) -> Optional[str]:
        """Name of the tool currently running, if in :attr:`Phase.ACTING`."""
        return self._active_tool

    @property
    def tool_count(self) -> int:
        """Number of tools started during the current turn."""
        return self._tool_count

    def begin_turn(self) -> None:
        """Reset per-turn state; the model call is now in flight."""
        self._phase = Phase.WAITING
        self._active_tool = None
        self._tool_count = 0

    def update(self, action: RenderAction) -> None:
        """Advance the inferred phase from one render action."""
        kind = action.kind
        if kind is RenderKind.THINKING:
            self._phase = Phase.PLANNING
        elif kind is RenderKind.TOOL_START:
            self._phase = Phase.ACTING
            self._active_tool = action.tool_name
            self._tool_count += 1
        elif kind is RenderKind.TOOL_END:
            # Stay in ACTING (more tools may follow); clear the active-tool name.
            self._active_tool = None
        elif kind is RenderKind.TOKEN:
            if action.text:
                self._phase = Phase.RESPONDING
        # ERROR / IGNORE do not change the phase.

    def end_turn(self) -> None:
        """Mark the turn complete."""
        self._phase = Phase.DONE
        self._active_tool = None

    def reset(self) -> None:
        """Return to the idle (no-turn) state."""
        self._phase = Phase.IDLE
        self._active_tool = None
        self._tool_count = 0

    def label(self) -> str:
        """Short, human-readable phase label for the status bar."""
        if self._phase is Phase.ACTING and self._active_tool:
            return f"acting · {self._active_tool}"
        return {
            Phase.IDLE: "idle",
            Phase.WAITING: "working…",
            Phase.PLANNING: "planning",
            Phase.ACTING: "acting",
            Phase.RESPONDING: "responding",
            Phase.DONE: "done",
        }[self._phase]
