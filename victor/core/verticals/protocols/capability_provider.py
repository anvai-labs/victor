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

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from victor.framework.persona_registry import PersonaSpec


# =============================================================================
# Capability Provider Protocol
# =============================================================================


@runtime_checkable
class CapabilityProviderProtocol(Protocol):
    """Protocol for capability configuration providers.

    Enables verticals to declare which capabilities they support,
    allowing runtime discovery and feature toggling.

    Example:
        class CodingCapabilityProvider(CapabilityProviderProtocol):
            def get_capabilities(self) -> Dict[str, Any]:
                return {
                    "code_review": True,
                    "refactoring": True,
                    "test_generation": True,
                    "max_file_size": 100000,
                }
    """

    def get_capabilities(self) -> Dict[str, Any]:
        """Get capability definitions for this vertical.

        Returns:
            Dict mapping capability names to their configurations
            (typically bool for feature flags, or values for limits)
        """
        ...


# =============================================================================
# Chain Provider Protocol
# =============================================================================


@runtime_checkable
class ChainProviderProtocol(Protocol):
    """Protocol for chain configuration providers.

    Enables verticals to define chains of operations that can be executed
    in sequence, supporting DIP by providing a protocol interface rather
    than concrete implementations.

    Example:
        class CodingChainProvider(ChainProviderProtocol):
            def get_chains(self) -> Dict[str, Any]:
                return {
                    "refactor": RefactorChain(steps=[...]),
                    "test_and_fix": TestFixChain(steps=[...]),
                }
    """

    def get_chains(self) -> Dict[str, Any]:
        """Get chain definitions for this vertical.

        Returns:
            Dict mapping chain names to chain configurations or instances
        """
        ...


# =============================================================================
# Persona Provider Protocol
# =============================================================================


@runtime_checkable
class PersonaProviderProtocol(Protocol):
    """Protocol for persona configuration providers.

    Enables verticals to define different personas (behavioral profiles)
    that affect how the agent interacts and responds.

    Example:
        class CodingPersonaProvider(PersonaProviderProtocol):
            def get_personas(self) -> Dict[str, Any]:
                return {
                    "senior_dev": {"name": "Senior Developer", "style": "thorough"},
                    "junior_dev": {"name": "Junior Developer", "style": "verbose"},
                }
    """

    def get_personas(self) -> Dict[str, Any]:
        """Get persona definitions for this vertical.

        Returns:
            Dict mapping persona names to persona configurations
        """
        ...


# =============================================================================
# Vertical Persona Provider Protocol (Enhanced)
# =============================================================================


@runtime_checkable
class VerticalPersonaProviderProtocol(Protocol):
    """Enhanced protocol for vertical-specific persona providers.

    Provides typed PersonaSpec objects and integrates with PersonaRegistry.
    Verticals implementing this protocol can contribute personas that are
    automatically registered in the global PersonaRegistry.

    Example:
        class CodingPersonaProvider(VerticalPersonaProviderProtocol):
            @classmethod
            def get_persona_specs(cls) -> Dict[str, "PersonaSpec"]:
                from victor.framework.persona_registry import PersonaSpec
                return {
                    "expert_reviewer": PersonaSpec(
                        name="expert_reviewer",
                        role="Expert Code Reviewer",
                        expertise=["code review", "security", "best practices"],
                        communication_style="professional",
                        behavioral_traits=["thorough", "constructive"],
                    ),
                    "mentor": PersonaSpec(
                        name="mentor",
                        role="Coding Mentor",
                        expertise=["teaching", "explaining"],
                        communication_style="supportive",
                    ),
                }

            @classmethod
            def get_default_persona(cls) -> Optional[str]:
                return "expert_reviewer"
    """

    @classmethod
    def get_persona_specs(cls) -> Dict[str, Any]:  # PersonaSpec
        """Get typed PersonaSpec objects for this vertical.

        Returns:
            Dict mapping persona names to PersonaSpec instances
        """
        ...

    @classmethod
    def get_default_persona(cls) -> Optional[str]:
        """Get the default persona name for this vertical.

        Returns:
            Name of the default persona, or None to use global default
        """
        ...

    @classmethod
    def get_persona_tags(cls) -> List[str]:
        """Get tags to apply to all personas from this vertical.

        Returns:
            List of tags (e.g., ["coding", "development"])
        """
        ...


__all__ = [
    "CapabilityProviderProtocol",
    "ChainProviderProtocol",
    "PersonaProviderProtocol",
    "VerticalPersonaProviderProtocol",
]
