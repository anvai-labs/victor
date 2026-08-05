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

"""Config Extensions Protocol (ISP: Interface Segregation Principle).

This module provides a focused protocol for configuration-related extensions
as part of the VerticalExtensions ISP refactoring.

Following the Interface Segregation Principle (ISP), this protocol contains
ONLY configuration-related fields extracted from the larger VerticalExtensions
interface. This allows verticals to implement only the config extensions they
need without being forced to depend on unrelated interfaces.

Usage:
    from victor.core.verticals.protocols.focused.config_extensions import (
        ConfigExtensionsProtocol,
    )

    class CodingConfigExtensions(ConfigExtensionsProtocol):
        def get_mode_configs(self) -> Dict[str, ModeConfig]:
            return {...}

Related Protocols:
    - ModeConfigProviderProtocol: The underlying protocol for mode configuration
    - ConfigExtensionsProtocol: Focused protocol for config-related extensions
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Dict, Protocol, runtime_checkable

from victor.core.verticals.protocols.mode_provider import (
    ModeConfig,
    ModeConfigProviderProtocol,
)

# =============================================================================
# Config Extensions Protocol
# =============================================================================


@runtime_checkable
class ConfigExtensionsProtocol(Protocol):
    """Protocol for configuration-related vertical extensions.

    This focused protocol contains ONLY configuration-related fields extracted
    from the larger VerticalExtensions interface. This follows the Interface
    Segregation Principle (ISP), allowing verticals to implement config
    extensions without depending on unrelated interfaces.

    The protocol provides access to:
    - mode_config_provider: Mode configuration provider for domain-specific
      operational modes (e.g., "fast", "thorough" for coding)

    Example:
        class CodingConfigExtensions(ConfigExtensionsProtocol):
            def __init__(self):
                self.mode_config_provider = CodingModeProvider()

            def get_all_mode_configs(self) -> Dict[str, ModeConfig]:
                if self.mode_config_provider:
                    return self.mode_config_provider.get_mode_configs()
                return {}
    """

    @property
    @abstractmethod
    def mode_config_provider(self) -> ModeConfigProviderProtocol | None:
        """Get the mode configuration provider.

        Returns:
            Mode configuration provider for domain-specific operational modes,
            or None if not configured.
        """
        ...

    def get_all_mode_configs(self) -> Dict[str, ModeConfig]:
        """Get mode configs from provider.

        Default implementation that delegates to the mode_config_provider.
        Subclasses may override for custom behavior.

        Returns:
            Dict of mode configurations
        """
        if self.mode_config_provider:
            return self.mode_config_provider.get_mode_configs()
        return {}


__all__ = [
    "ConfigExtensionsProtocol",
]
