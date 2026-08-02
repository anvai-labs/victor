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

"""Per-session ratchet state for the proactive spin-escape modifier (ADR-013).

The ratchet *count* is advanced ONLY by :meth:`RatchetState.record_turn` (called once per turn by the
loop), never by :meth:`SpinRatchetModifier.adjust`. This keeps ``adjust`` a pure function of
``(value, steps)`` — idempotent and safe to call repeatedly for tracing without double-stepping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from victor.framework.temperature.protocols import SpinSignal

# Spin states (SpinState.value) that justify escalating temperature.
_RATCHET_STATES = frozenset({"warning", "stuck"})


@dataclass
class RatchetState:
    """Accumulated escalation steps for one session. Reset on progress."""

    steps: int = 0

    def record_turn(self, spin: SpinSignal) -> None:
        """Advance/reset the ratchet for a completed turn (call exactly once per turn)."""
        if spin.made_progress:
            self.steps = 0
        elif spin.spin_state in _RATCHET_STATES or spin.plateaued:
            self.steps += 1

    def reset(self) -> None:
        self.steps = 0


@dataclass
class RatchetStateRegistry:
    """Maps ``session_id`` → :class:`RatchetState` so a singleton resolver stays stateless.

    The loop calls :meth:`record_turn` once per turn; the resolver reads the per-session state via
    ``TemperatureContext.ratchet_state``. Drop a session with :meth:`discard` on cleanup.
    """

    _states: Dict[str, RatchetState] = field(default_factory=dict)

    def get_or_create(self, session_id: str) -> RatchetState:
        state = self._states.get(session_id)
        if state is None:
            state = RatchetState()
            self._states[session_id] = state
        return state

    def record_turn(self, session_id: str, spin: SpinSignal) -> RatchetState:
        state = self.get_or_create(session_id)
        state.record_turn(spin)
        return state

    def discard(self, session_id: str) -> None:
        self._states.pop(session_id, None)
