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

"""Agentic-loop spin guards and telemetry.

A real bug: on a real repo, the agentic loop ran ~5.5 hours effectively spinning
while the visible iteration bound (``for i in range(1, effective_max+1)``) looked
respected. Diagnosis showed a single perception call is fast (~300ms), so the
unbounded axis is NOT the iteration counter — it is the *number of perception
calls* (perception is invoked exactly once per iteration in healthy runs, so
``perception_calls >> iteration_count`` is an unambiguous restart/re-entry spin
signal that the iteration counter and the post-ACT ``SpinDetector`` both miss).

``LoopGuards`` is a per-run counter + monotonic timer object, threaded to the
perception seam via a :class:`contextvars.ContextVar` so both the legacy loop
and the StateGraph loop are covered without signature churn. This module is
log-only in Phase 1 (instrumentation); the terminating deadline/backstop land in
later phases (they reuse these same counters).

Kept out of ``agentic_loop.py`` (a large hotspot) on purpose — the loop only
gains thin call-sites.
"""

from __future__ import annotations

import contextvars
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Healthy runs call perception exactly once per iteration. This many times the
# iteration count is a strong spin signal without false positives on legitimate
# re-perception (e.g. one retry per turn would still be ~2x).
PERCEPTION_SPIN_FACTOR = 4

# The current run's guards, so the shared perception seam can find them without
# threading a parameter through every call site. Unset (None) outside a loop.
_CURRENT_GUARDS: "contextvars.ContextVar[Optional[LoopGuards]]" = contextvars.ContextVar(
    "victor_loop_guards", default=None
)


@dataclass
class LoopGuards:
    """Per-run spin telemetry: perception/iteration/restart counters + timers.

    Phase 1 is log-only (rate-limited WARNING when a threshold trips). The
    terminating deadline/backstop (later phases) read the same fields.
    """

    perception_spin_factor: int = PERCEPTION_SPIN_FACTOR
    perception_calls: int = 0
    iteration_count: int = 0
    restart_count: int = 0
    run_started_monotonic: float = field(default_factory=time.monotonic)
    phase_started_monotonic: float = field(default_factory=time.monotonic)
    _warned_spin: bool = field(default=False, repr=False)

    # ----- counters -----

    def note_iteration(self) -> None:
        self.iteration_count += 1

    def note_restart(self) -> None:
        self.restart_count += 1

    def note_perception(self) -> None:
        """Count a perception call; warn once if it far outpaces iterations."""
        self.perception_calls += 1
        threshold = max(self.iteration_count, 1) * self.perception_spin_factor
        if self.perception_calls > threshold and not self._warned_spin:
            self._warned_spin = True
            logger.warning(
                "[LoopGuards] perception_calls=%d >> iterations=%d (restarts=%d) — "
                "possible perception/restart spin: perception is called once per "
                "iteration in healthy runs. total_elapsed=%.1fs",
                self.perception_calls,
                self.iteration_count,
                self.restart_count,
                self.total_elapsed(),
            )

    # ----- timers -----

    def start_phase(self) -> None:
        self.phase_started_monotonic = time.monotonic()

    def phase_elapsed(self) -> float:
        return time.monotonic() - self.phase_started_monotonic

    def total_elapsed(self) -> float:
        return time.monotonic() - self.run_started_monotonic

    def check_phase(self, phase: str, soft_cap_s: Optional[float]) -> bool:
        """Warn (once-ish) if a single phase exceeds ``soft_cap_s``. Returns breach."""
        if soft_cap_s is None:
            return False
        elapsed = self.phase_elapsed()
        if elapsed > soft_cap_s:
            logger.warning(
                "[LoopGuards] phase %r took %.1fs (soft cap %.1fs) at iteration %d",
                phase,
                elapsed,
                soft_cap_s,
                self.iteration_count,
            )
            return True
        return False

    def check_total(self, deadline_s: Optional[float]) -> bool:
        """True if total wall-clock exceeded ``deadline_s`` (log-only in Phase 1)."""
        if deadline_s is None:
            return False
        if self.total_elapsed() > deadline_s:
            logger.warning(
                "[LoopGuards] loop wall-clock %.1fs exceeded deadline %.1fs "
                "(iterations=%d, perception_calls=%d)",
                self.total_elapsed(),
                deadline_s,
                self.iteration_count,
                self.perception_calls,
            )
            return True
        return False


def set_current_guards(guards: Optional[LoopGuards]) -> "contextvars.Token":
    """Install ``guards`` as the current run's guards; returns a reset token."""
    return _CURRENT_GUARDS.set(guards)


def reset_current_guards(token: "contextvars.Token") -> None:
    _CURRENT_GUARDS.reset(token)


def current_guards() -> Optional["LoopGuards"]:
    """The current run's guards, or None outside a guarded loop."""
    return _CURRENT_GUARDS.get()


def note_perception() -> None:
    """Count a perception call on the current run's guards (no-op if unset).

    Called from the shared perception seam so both loop implementations are
    covered by one hook.
    """
    guards = _CURRENT_GUARDS.get()
    if guards is not None:
        guards.note_perception()
