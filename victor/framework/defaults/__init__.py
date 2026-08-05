# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Framework defaults — one-stop import for external verticals.

Collects generic stage definitions, safety patterns, RL config, mode
config, persona helpers, and task hints so that external verticals can
bootstrap with::

    from victor.framework.defaults import (
        get_default_stages,
        DefaultSafetyExtension,
        BaseRLConfig,
        DEFAULT_COMPLEXITY_MAP,
        PersonaHelpers,
        MODIFICATION_STAGE_TOOLS,
        scale_budget,
    )
"""

from __future__ import annotations

# Stages
from victor.framework.defaults.stages import get_default_stages

# Safety
from victor.framework.defaults.safety import DefaultSafetyExtension

# RL (re-export from existing)
from victor.framework.rl.config import (
    BaseRLConfig,
    DEFAULT_ACTIVE_LEARNERS,
    DEFAULT_PATIENCE_MAP,
)

# Mode config (re-export from existing)
from victor.core.mode_config import (
    DEFAULT_MODES,
    DEFAULT_TASK_BUDGETS,
    ModeConfigRegistry,
    ModeDefinition,
    RegistryBasedModeConfigProvider,
)

# Stage definition type for external/default contracts
from victor_contracts import StageDefinition

# Mode configs (defaults)
from victor.framework.defaults.mode_configs import (
    DEFAULT_COMPLEXITY_MAP,
    create_complexity_map,
)

# Personas
from victor.framework.defaults.personas import PersonaHelpers

# Task hints
from victor.framework.defaults.task_hints import (
    ANALYSIS_STAGE_TOOLS,
    DEFAULT_BUDGET_SCALING,
    EXECUTION_STAGE_TOOLS,
    MODIFICATION_STAGE_TOOLS,
    get_stage_tools_for_category,
    scale_budget,
)

__all__ = [
    # Stages
    "get_default_stages",
    "StageDefinition",
    # Safety
    "DefaultSafetyExtension",
    # RL
    "BaseRLConfig",
    "DEFAULT_ACTIVE_LEARNERS",
    "DEFAULT_PATIENCE_MAP",
    # Mode config
    "ModeConfigRegistry",
    "ModeDefinition",
    "RegistryBasedModeConfigProvider",
    "DEFAULT_MODES",
    "DEFAULT_TASK_BUDGETS",
    # Mode configs (defaults)
    "DEFAULT_COMPLEXITY_MAP",
    "create_complexity_map",
    # Personas
    "PersonaHelpers",
    # Task hints
    "MODIFICATION_STAGE_TOOLS",
    "ANALYSIS_STAGE_TOOLS",
    "EXECUTION_STAGE_TOOLS",
    "get_stage_tools_for_category",
    "DEFAULT_BUDGET_SCALING",
    "scale_budget",
]
