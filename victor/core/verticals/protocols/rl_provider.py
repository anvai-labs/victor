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

"""RL Provider Protocols (ISP: Interface Segregation Principle).

This module contains protocols specifically for RL (Reinforcement Learning)
configuration. Following ISP, these protocols are focused on a single
responsibility: providing RL learner configurations.

Usage:
    from victor.core.verticals.protocols.rl_provider import (
        RLConfigProviderProtocol,
        VerticalRLProviderProtocol,
    )

    class CodingRLConfigProvider(RLConfigProviderProtocol):
        def get_rl_config(self) -> Dict[str, Any]:
            return {
                "active_learners": ["tool_selection", "semantic_threshold"],
                "quality_thresholds": {"code_review": 0.8, "bugfix": 0.85},
            }
"""

from __future__ import annotations

# The definitions below live once in the SDK (promoted from this module) and
# are re-exported here as the SAME objects, so ``isinstance`` checks and
# dataclass identity agree with ``victor.core.verticals.protocols`` and
# ``victor_contracts.verticals.protocols``.
from victor_contracts.verticals.protocols.promoted import (
    RLConfigProviderProtocol as RLConfigProviderProtocol,
    VerticalRLProviderProtocol as VerticalRLProviderProtocol,
)

__all__ = [
    "RLConfigProviderProtocol",
    "VerticalRLProviderProtocol",
]
