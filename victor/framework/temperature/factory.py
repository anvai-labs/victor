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

"""Assembly authority for the default :class:`TemperatureResolver` (ADR-013).

The single place that fixes source precedence and modifier order. Adding a strategy is an edit here,
not in the resolver (OCP). Settings/DI wiring (PR-B) calls this with values from
``Settings.temperature``; for tests it is callable with explicit args and no settings dependency.
"""

from __future__ import annotations

from typing import Optional

from victor.framework.temperature.defaults import (
    DEFAULT_RATCHET_CAP,
    DEFAULT_RATCHET_STEP,
    GLOBAL_DEFAULT,
)
from victor.framework.temperature.modifiers import (
    ModelBoundsModifier,
    SpinRatchetModifier,
)
from victor.framework.temperature.protocols import ReactiveTemperatureAdjuster
from victor.framework.temperature.recovery_modifier import RecoveryAdjustModifier
from victor.framework.temperature.resolver import TemperatureResolver
from victor.framework.temperature.sources import (
    GlobalDefaultSource,
    ProfileBaseSource,
    ProfilePerTaskSource,
    SettingsPerTaskSource,
    TaskHintConstantSource,
)


def build_default_resolver(
    *,
    hint_provider: Optional[object] = None,
    reactive_adjuster: Optional[ReactiveTemperatureAdjuster] = None,
    global_default: float = GLOBAL_DEFAULT,
    ratchet_step: float = DEFAULT_RATCHET_STEP,
    ratchet_cap: float = DEFAULT_RATCHET_CAP,
    ratchet_enabled: bool = True,
) -> TemperatureResolver:
    """Build the standard resolver: precedence-ordered sources + ratchet→recovery→bounds modifiers."""
    sources = [
        ProfilePerTaskSource(),
        SettingsPerTaskSource(),
        TaskHintConstantSource(hint_provider),
        ProfileBaseSource(),
        GlobalDefaultSource(global_default),
    ]

    modifiers = [
        SpinRatchetModifier(step=ratchet_step, cap=ratchet_cap, enabled=ratchet_enabled),
    ]
    if reactive_adjuster is not None:
        modifiers.append(RecoveryAdjustModifier(reactive_adjuster))
    modifiers.append(ModelBoundsModifier())

    return TemperatureResolver(sources, modifiers, global_default=global_default)


def build_resolver_from_settings(
    temperature_settings: object,
    *,
    hint_provider: Optional[object] = None,
    reactive_adjuster: Optional[ReactiveTemperatureAdjuster] = None,
) -> TemperatureResolver:
    """Build a resolver from a ``TemperatureSettings``-shaped object (duck-typed, no config import).

    The per-task ``task_defaults`` table is threaded per-request (``settings_task_temperatures``), not
    at build time, so only the global/ratchet knobs are read here.
    """
    ts = temperature_settings
    return build_default_resolver(
        hint_provider=hint_provider,
        reactive_adjuster=reactive_adjuster,
        global_default=getattr(ts, "global_default", GLOBAL_DEFAULT),
        ratchet_step=getattr(ts, "ratchet_step", DEFAULT_RATCHET_STEP),
        ratchet_cap=getattr(ts, "ratchet_cap", DEFAULT_RATCHET_CAP),
        ratchet_enabled=getattr(ts, "proactive_ratchet_enabled", True),
    )
