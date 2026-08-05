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

"""Temperature modifiers (ADR-013), applied in order ``ratchet → recovery → bounds``.

Each ``adjust`` is idempotent for identical inputs. The proactive ratchet reads an externally-advanced
step count (see :mod:`ratchet_state`), so calling ``adjust`` repeatedly never double-steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from victor.core.utils import clamp
from victor.framework.temperature.defaults import (
    DEFAULT_RATCHET_CAP,
    DEFAULT_RATCHET_STEP,
    escalate_temperature,
    model_bounds,
)
from victor.framework.temperature.protocols import (
    TemperatureContext,
    TemperatureRequest,
)


@dataclass
class SpinRatchetModifier:
    """Proactive escape velocity: raise temperature while a session is spinning/plateaued.

    ``value += steps * step`` (steps from the per-session :class:`RatchetState` on the context),
    hard-capped at ``cap`` (kept below the degeneration cliff, arXiv 2606.01451). Reset is handled by
    ``RatchetState.record_turn`` on progress — not here — preserving idempotency.
    """

    step: float = DEFAULT_RATCHET_STEP
    cap: float = DEFAULT_RATCHET_CAP
    enabled: bool = True

    @property
    def name(self) -> str:
        return "spin_ratchet"

    def adjust(
        self, value: float, request: TemperatureRequest, context: TemperatureContext
    ) -> Tuple[float, str]:
        if not self.enabled:
            return value, "ratchet disabled"
        state = context.ratchet_state
        steps = int(getattr(state, "steps", 0) or 0)
        if steps <= 0:
            return value, "no ratchet (steps=0)"
        new_value = escalate_temperature(value, steps * self.step, cap=self.cap)
        return (
            new_value,
            f"ratchet +{new_value - value:.3f} (steps={steps}, cap={self.cap})",
        )


@dataclass
class ModelBoundsModifier:
    """Terminal clamp to per-model effective bounds — nothing escapes provider limits."""

    @property
    def name(self) -> str:
        return "model_bounds"

    def adjust(
        self, value: float, request: TemperatureRequest, context: TemperatureContext
    ) -> Tuple[float, str]:
        low, high = model_bounds(request.model_name)
        clamped = clamp(value, low, high)
        if clamped != value:
            return (
                clamped,
                f"clamped to [{low}, {high}] for {request.model_name or 'unknown'}",
            )
        return value, f"within [{low}, {high}]"
