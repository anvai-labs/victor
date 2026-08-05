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

"""Reactive recovery modifier (ADR-013) — absorbs the existing failure escalation.

Wraps an injected :class:`ReactiveTemperatureAdjuster` (structurally satisfied by the agent-layer
``ProgressiveTemperatureAdjuster``: failure-type policies + Q-learning + per-model bounds). When a
reactive failure is active for the turn (``context.recovery_context`` set) it delegates; otherwise it
passes the value through. This unifies the previously-parallel recovery escalation path into the one
resolver rather than reimplementing it (the same adjuster instance is shared via DI, so Q-learning
state stays consistent with the recovery coordinator).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from victor.framework.temperature.protocols import (
    ReactiveTemperatureAdjuster,
    TemperatureContext,
    TemperatureRequest,
)


@dataclass
class RecoveryAdjustModifier:
    """Delegate to the reactive adjuster when a recovery context is present; else passthrough."""

    adjuster: ReactiveTemperatureAdjuster

    @property
    def name(self) -> str:
        return "recovery_adjust"

    def adjust(
        self, value: float, request: TemperatureRequest, context: TemperatureContext
    ) -> Tuple[float, str]:
        if context.recovery_context is None:
            return value, "no active recovery"
        new_value, reason = self.adjuster.get_adjusted_temperature(
            context.recovery_context, context.session_id
        )
        return new_value, f"recovery: {reason}"
