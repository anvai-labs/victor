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

"""Shared temperature-resolution seam for the buffered and streaming paths (ADR-013).

Both ``TurnExecutor`` (buffered) and the streaming pipeline call :func:`resolve_effective_temperature`
so a given ``(task_type, model, base, context)`` resolves identically regardless of mode — this is what
closes the buffered/streaming divergence. The helper is orchestrator-duck-typed (reads
``temperature_resolver`` / ``profile_task_temperatures`` / ``settings_task_temperatures``) and degrades
to ``base_temperature`` when no resolver is available.
"""

from __future__ import annotations

from typing import Any, Optional

from victor.framework.temperature import TemperatureContext, TemperatureRequest


def normalize_task_type(value: Any) -> Optional[str]:
    """Coerce a task-type-ish value (str / enum / None) to a string key, or None."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return getattr(value, "value", None)


def resolve_effective_temperature(
    orchestrator: Any,
    *,
    task_type: Optional[str],
    model: str,
    base_temperature: Optional[float],
    explicit_override: Optional[float] = None,
    context: Optional[TemperatureContext] = None,
) -> Optional[float]:
    """Resolve the effective sampling temperature via the orchestrator-owned resolver (ADR-013).

    An explicit caller override (heterogeneous teams / recovery) wins. Otherwise build a
    :class:`TemperatureRequest` from the orchestrator's profile/settings per-task maps and delegate to
    its :class:`TemperatureResolver`; fall back to ``base_temperature`` if no resolver is available.
    """
    if explicit_override is not None:
        return explicit_override
    resolver = getattr(orchestrator, "temperature_resolver", None) if orchestrator else None
    if resolver is None:
        return base_temperature
    if context is None:
        # Auto-attach the per-session ratchet state (ADR-013, PR-E) so the SpinRatchetModifier sees
        # accumulated escape-velocity steps. The loop advances it once per turn; we only read here.
        ratchet = getattr(orchestrator, "temperature_ratchet_state", None)
        if ratchet is not None:
            context = TemperatureContext(ratchet_state=ratchet)
    request = TemperatureRequest(
        # Normalize: callers may pass a TaskType enum (perception) or a string; the resolver's
        # task-hint lookup needs a plain string (it calls .lower()). None → "default".
        task_type=normalize_task_type(task_type) or "default",
        model_name=model or "",
        profile_temperature=base_temperature,
        profile_task_temperatures=getattr(orchestrator, "profile_task_temperatures", {}) or {},
        settings_task_temperatures=getattr(orchestrator, "settings_task_temperatures", {}) or {},
    )
    return resolver.resolve(request, context).value
