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

"""Traits + value objects for the unified temperature policy (ADR-013).

Two narrow Protocols (ISP) drive composition in :class:`TemperatureResolver`:

- :class:`TemperatureSource` resolves a *base* temperature from static inputs only
  (:class:`TemperatureRequest`) — pure, no spin/IO — returning ``None`` to defer to the next source.
- :class:`TemperatureModifier` *adjusts* a resolved value, additionally seeing the dynamic
  :class:`TemperatureContext` (spin/recovery). ``adjust`` MUST be idempotent for identical inputs.

The split keeps a source from accidentally depending on spin, and a modifier from re-resolving a base.
:class:`ReactiveTemperatureAdjuster` inverts the framework→agent dependency for wrapping the existing
``ProgressiveTemperatureAdjuster`` (recovery) without importing it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, Tuple, runtime_checkable


@dataclass(frozen=True)
class TemperatureRequest:
    """Immutable static inputs to resolve a *base* temperature (no spin/IO)."""

    task_type: str = "default"
    provider_name: str = ""
    model_name: str = ""
    profile_temperature: Optional[float] = None
    profile_task_temperatures: Dict[str, float] = field(default_factory=dict)
    settings_task_temperatures: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SpinSignal:
    """Per-turn spin/plateau snapshot for the proactive ratchet (read from SpinDetector/PlateauDetector)."""

    spin_state: str = "normal"  # SpinState.value
    consecutive_no_tool_turns: int = 0
    plateaued: bool = False
    made_progress: bool = False  # fulfillment / new tool evidence this turn → reset trigger


@dataclass(frozen=True)
class TemperatureContext:
    """Per-turn dynamic context for modifiers (kept separate from the static request by ISP)."""

    session_id: Optional[str] = None
    spin: SpinSignal = field(default_factory=SpinSignal)
    ratchet_state: Optional["object"] = None  # RatchetState; opaque here to avoid a cycle
    recovery_context: Optional[Any] = None  # RecoveryContext when a reactive failure is active


@dataclass(frozen=True)
class TemperatureResolution:
    """Final resolved temperature plus an audit trail (observability)."""

    value: float
    base: float
    source_name: str
    modifier_trace: Tuple[Tuple[str, float, str], ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": round(self.value, 4),
            "base": round(self.base, 4),
            "source": self.source_name,
            "modifiers": [
                {"name": n, "value": round(v, 4), "reason": r} for (n, v, r) in self.modifier_trace
            ],
        }


@runtime_checkable
class TemperatureSource(Protocol):
    """Resolves a base temperature, or ``None`` to defer to the next source (Chain of Responsibility)."""

    @property
    def name(self) -> str: ...

    def resolve(self, request: TemperatureRequest) -> Optional[float]: ...


@runtime_checkable
class TemperatureModifier(Protocol):
    """Adjusts a resolved temperature. ``adjust`` must be idempotent for identical inputs."""

    @property
    def name(self) -> str: ...

    def adjust(
        self, value: float, request: TemperatureRequest, context: TemperatureContext
    ) -> Tuple[float, str]: ...


@runtime_checkable
class ReactiveTemperatureAdjuster(Protocol):
    """Narrow inversion of the agent-layer recovery adjuster (``ProgressiveTemperatureAdjuster``).

    Lets framework code wrap reactive failure-escalation without importing ``victor.agent``.
    """

    def get_adjusted_temperature(
        self, context: Any, session_id: Optional[str] = None
    ) -> Tuple[float, str]: ...
