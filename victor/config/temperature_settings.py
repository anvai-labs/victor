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

"""Settings for the unified temperature policy (ADR-013).

Ops-tunable knobs for ``victor.framework.temperature``: the global fallback, a per-task default table
(an alternative to per-profile ``temperatures``), and the spin-ratchet parameters. Defaults match
``framework/temperature/defaults`` so behaviour is identical whether or not this group is customised.
"""

from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, Field


class TemperatureSettings(BaseModel):
    """Unified temperature-policy configuration (sampling temperature)."""

    # Terminal base when no source (profile-per-task, settings-per-task, task-hint, profile-base)
    # resolves one. 0.6 favours determinism with ratchet headroom (ADR-013).
    global_default: float = Field(0.6, ge=0.0, le=2.0)

    # Ops-level per-task overrides (task_type -> temperature). Sits above the TaskTypeHint constants
    # and below per-profile `temperatures`. Empty by default → defer to the constant floor.
    task_defaults: Dict[str, float] = Field(default_factory=dict)

    # Proactive spin-escape ratchet (arXiv 2606.01451: modest step, cap below the ~1.0 cliff).
    proactive_ratchet_enabled: bool = True
    ratchet_step: float = Field(0.05, ge=0.0, le=0.5)
    ratchet_cap: float = Field(0.9, ge=0.0, le=2.0)

    # Reserved: lower the ratchet cap for short generation budgets (budget-aware, ANTS 2606.13982).
    budget_aware_cap: bool = True
