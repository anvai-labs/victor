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

"""Unified sampling-temperature policy (ADR-013).

One composition-based resolver that all call sites route through: intent/task-based base temperature
resolved ``profile-per-task › settings-per-task › task-hint constant › profile base › global default``,
plus composable modifiers (proactive spin ratchet, reactive recovery escalation, model-bounds clamp).

NOTE: this is *sampling* temperature — distinct from ``victor.agent.context_temperature`` (HYVE
message-tier temperature).
"""

from victor.framework.temperature.defaults import (
    DEFAULT_RATCHET_CAP,
    DEFAULT_RATCHET_STEP,
    GLOBAL_DEFAULT,
    MODEL_TEMPERATURE_RANGES,
    escalate_temperature,
    model_bounds,
)
from victor.framework.temperature.factory import (
    build_default_resolver,
    build_resolver_from_settings,
)
from victor.framework.temperature.modifiers import (
    ModelBoundsModifier,
    SpinRatchetModifier,
)
from victor.framework.temperature.protocols import (
    ReactiveTemperatureAdjuster,
    SpinSignal,
    TemperatureContext,
    TemperatureModifier,
    TemperatureRequest,
    TemperatureResolution,
    TemperatureSource,
)
from victor.framework.temperature.ratchet_state import (
    RatchetState,
    RatchetStateRegistry,
)
from victor.framework.temperature.recovery_modifier import RecoveryAdjustModifier
from victor.framework.temperature.resolver import TemperatureResolver
from victor.framework.temperature.sources import (
    GlobalDefaultSource,
    ProfileBaseSource,
    ProfilePerTaskSource,
    SettingsPerTaskSource,
    TaskHintConstantSource,
)

__all__ = [
    # defaults
    "GLOBAL_DEFAULT",
    "DEFAULT_RATCHET_STEP",
    "DEFAULT_RATCHET_CAP",
    "MODEL_TEMPERATURE_RANGES",
    "model_bounds",
    "escalate_temperature",
    # protocols / value objects
    "TemperatureRequest",
    "TemperatureContext",
    "TemperatureResolution",
    "SpinSignal",
    "TemperatureSource",
    "TemperatureModifier",
    "ReactiveTemperatureAdjuster",
    # sources
    "ProfilePerTaskSource",
    "SettingsPerTaskSource",
    "TaskHintConstantSource",
    "ProfileBaseSource",
    "GlobalDefaultSource",
    # modifiers
    "SpinRatchetModifier",
    "ModelBoundsModifier",
    "RecoveryAdjustModifier",
    # state
    "RatchetState",
    "RatchetStateRegistry",
    # resolver + factory
    "TemperatureResolver",
    "build_default_resolver",
    "build_resolver_from_settings",
]
