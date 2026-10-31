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

"""Mode Provider Protocols (ISP: Interface Segregation Principle).

This module contains protocols specifically for mode configuration.
Following ISP, these protocols are focused on a single responsibility:
providing operational mode configurations.

Usage:
    from victor.core.verticals.protocols.mode_provider import (
        ModeConfigProviderProtocol,
        ModeConfig,
    )

    class CodingModeProvider(ModeConfigProviderProtocol):
        def get_mode_configs(self) -> Dict[str, ModeConfig]:
            return {
                "fast": ModeConfig(name="fast", tool_budget=5, max_iterations=10),
                "thorough": ModeConfig(name="thorough", tool_budget=30, max_iterations=60),
            }
"""

from __future__ import annotations

# The definitions below live once in the SDK (promoted from this module) and
# are re-exported here as the SAME objects, so ``isinstance`` checks and
# dataclass identity agree with ``victor.core.verticals.protocols`` and
# ``victor_contracts.verticals.protocols``.
from victor_contracts.verticals.protocols.promoted import (
    ModeConfigProviderProtocol as ModeConfigProviderProtocol,
)
from victor_contracts.verticals.protocols.promoted_types import (
    ModeConfig as ModeConfig,
)

__all__ = [
    "ModeConfig",
    "ModeConfigProviderProtocol",
]
