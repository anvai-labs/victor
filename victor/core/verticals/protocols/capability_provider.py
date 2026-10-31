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

"""Capability Provider Protocols (ISP: Interface Segregation Principle).

This module contains protocols for capability, chain, and persona providers.
Following ISP, these protocols are focused on declaring vertical capabilities.

Usage:
    from victor.core.verticals.protocols.capability_provider import (
        CapabilityProviderProtocol,
        ChainProviderProtocol,
        PersonaProviderProtocol,
        VerticalPersonaProviderProtocol,
    )

    class CodingCapabilityProvider(CapabilityProviderProtocol):
        def get_capabilities(self) -> Dict[str, Any]:
            return {
                "code_review": True,
                "refactoring": True,
                "max_file_size": 100000,
            }

    class CodingPersonaProvider(VerticalPersonaProviderProtocol):
        @classmethod
        def get_persona_specs(cls) -> Dict[str, PersonaSpec]:
            from victor.framework.persona_registry import PersonaSpec
            return {
                "expert_reviewer": PersonaSpec(
                    name="expert_reviewer",
                    role="Expert Code Reviewer",
                    expertise=["code review", "security"],
                ),
            }
"""

from __future__ import annotations

# The definitions below live once in the SDK (promoted from this module) and
# are re-exported here as the SAME objects, so ``isinstance`` checks and
# dataclass identity agree with ``victor.core.verticals.protocols`` and
# ``victor_contracts.verticals.protocols``.
from victor_contracts.verticals.protocols.promoted import (
    CapabilityProviderProtocol as CapabilityProviderProtocol,
    ChainProviderProtocol as ChainProviderProtocol,
    PersonaProviderProtocol as PersonaProviderProtocol,
    VerticalPersonaProviderProtocol as VerticalPersonaProviderProtocol,
)

__all__ = [
    "CapabilityProviderProtocol",
    "ChainProviderProtocol",
    "PersonaProviderProtocol",
    "VerticalPersonaProviderProtocol",
]
