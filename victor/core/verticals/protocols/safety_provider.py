# Copyright 2025 Vijaykumar Singh <vijaykumar@anvaiops.com>
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

"""Safety Provider Protocols (ISP: Interface Segregation Principle).

This module contains protocols specifically for safety pattern management.
Following ISP, these protocols are focused on a single responsibility:
defining and providing dangerous operation patterns.

Usage:
    from victor.core.verticals.protocols.safety_provider import (
        SafetyExtensionProtocol,
    )

    class GitSafetyExtension(SafetyExtensionProtocol):
        def get_bash_patterns(self) -> List[SafetyPattern]:
            return [
                SafetyPattern(
                    pattern=r"git\\s+reset\\s+--hard",
                    description="Discard uncommitted changes",
                    risk_level="HIGH",
                    category="git",
                ),
            ]
"""

from __future__ import annotations

# The definitions below live once in the SDK (promoted from this module) and
# are re-exported here as the SAME objects, so ``isinstance`` checks and
# dataclass identity agree with ``victor.core.verticals.protocols`` and
# ``victor_contracts.verticals.protocols``.
from victor_contracts.verticals.protocols.promoted import (
    SafetyExtensionProtocol as SafetyExtensionProtocol,
)

# SafetyPattern stays a core dataclass re-export (the SDK spelling is
# SafetyPatternData); implementers returning the core dataclass satisfy the
# protocol identically at runtime.
from victor.security.safety.types import SafetyPattern

__all__ = [
    "SafetyExtensionProtocol",
    "SafetyPattern",
]
