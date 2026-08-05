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

"""Base-temperature sources (ADR-013), highest → lowest precedence.

Each source is a single-responsibility ``TemperatureSource``: it returns a base temperature for the
request's ``task_type`` or ``None`` to defer. Ordered by :func:`build_default_resolver`:
``ProfilePerTask › SettingsPerTask › TaskHintConstant › ProfileBase › GlobalDefault``.

Sources are pure — all data (profile/settings maps, profile base) arrives on the
:class:`TemperatureRequest`; the only injected collaborator is the in-memory task-hint provider.
"""

from __future__ import annotations

from typing import Any, Optional

from victor.framework.temperature.defaults import GLOBAL_DEFAULT
from victor.framework.temperature.protocols import TemperatureRequest


class ProfilePerTaskSource:
    """Per-task override from the active profile's ``temperatures`` map (e.g. ``glm5.2: {plan: 0.5}``)."""

    @property
    def name(self) -> str:
        return "profile_per_task"

    def resolve(self, request: TemperatureRequest) -> Optional[float]:
        return request.profile_task_temperatures.get(request.task_type)


class SettingsPerTaskSource:
    """Per-task override from the settings-level ops table (``temperature.task_defaults``)."""

    @property
    def name(self) -> str:
        return "settings_per_task"

    def resolve(self, request: TemperatureRequest) -> Optional[float]:
        return request.settings_task_temperatures.get(request.task_type)


class TaskHintConstantSource:
    """Per-task constant from ``TaskTypeHint.temperature_override`` (the SDK-stable constant floor).

    Reads the existing ``TaskTypeHintCapabilityProvider`` (debug 0.1 … analyze 0.6) — the constants
    stay in their canonical home; this source merely reads them, so nothing is duplicated.
    """

    def __init__(self, hint_provider: Any) -> None:
        self._provider = hint_provider

    @property
    def name(self) -> str:
        return "task_hint_constant"

    def resolve(self, request: TemperatureRequest) -> Optional[float]:
        if self._provider is None:
            return None
        hint = self._provider.get_hint(request.task_type)
        return getattr(hint, "temperature_override", None) if hint is not None else None


class ProfileBaseSource:
    """The profile's single base temperature (``ProfileConfig.temperature``)."""

    @property
    def name(self) -> str:
        return "profile_base"

    def resolve(self, request: TemperatureRequest) -> Optional[float]:
        return request.profile_temperature


class GlobalDefaultSource:
    """Terminal source: always returns the configured global default (resolution is total)."""

    def __init__(self, default: float = GLOBAL_DEFAULT) -> None:
        self._default = default

    @property
    def name(self) -> str:
        return "global_default"

    def resolve(self, request: TemperatureRequest) -> Optional[float]:
        return self._default
