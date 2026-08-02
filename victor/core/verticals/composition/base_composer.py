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

"""Base composer for vertical capabilities.

Provides the foundation for composing vertical capabilities
in a declarative, builder-style API.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Type

if TYPE_CHECKING:
    from victor.core.verticals.base import VerticalBase

logger = logging.getLogger(__name__)


class BaseCapabilityProvider:
    """Base class for capability providers.

    Capability providers encapsulate specific vertical capabilities
    like metadata, stages, extensions, workflows, etc.

    Each provider should follow the Single Responsibility Principle
    by handling only one specific capability.

    Example:
        class MetadataCapability(BaseCapabilityProvider):
            def __init__(self, name, description, version):
                self.name = name
                self.description = description
                self.version = version
    """

    def __init__(self):
        self._config: Dict[str, Any] = {}

    def get_config(self) -> Dict[str, Any]:
        """Get the capability configuration.

        Returns:
            Configuration dictionary for this capability
        """
        return self._config.copy()

    def validate(self) -> bool:
        """Validate the capability configuration.

        Returns:
            True if configuration is valid
        """
        return True


class BaseComposer:
    """Base class for composing vertical capabilities.

    Provides the foundational composition API that can be extended
    by more specific composers like CapabilityComposer.

    Example:
        composer = BaseComposer(MyVertical)
        composer.register_capability("metadata", MetadataCapability(...))
    """

    def __init__(self, vertical: Type["VerticalBase"]):
        """Initialize the composer.

        Args:
            vertical: The vertical base class to compose
        """
        self._vertical = vertical
        self._capability_providers: Dict[str, BaseCapabilityProvider] = {}
        self._logger = logging.getLogger(f"{__name__}.{id(self)}")

    def register_capability(
        self,
        name: str,
        provider: BaseCapabilityProvider,
    ) -> "BaseComposer":
        """Register a capability provider.

        Note: Validation is performed during build(), not during registration.
        This allows registering capabilities before they're fully configured.

        Args:
            name: Name of the capability
            provider: Capability provider instance

        Returns:
            Self for method chaining
        """
        self._capability_providers[name] = provider
        self._logger.debug(f"Registered capability: {name}")

        return self

    def get_capability(
        self,
        name: str,
    ) -> Optional[BaseCapabilityProvider]:
        """Get a capability provider.

        Args:
            name: Name of the capability

        Returns:
            Capability provider if found, None otherwise
        """
        return self._capability_providers.get(name)

    def has_capability(self, name: str) -> bool:
        """Check if a capability is registered.

        Args:
            name: Name of the capability

        Returns:
            True if capability is registered
        """
        return name in self._capability_providers

    def list_capabilities(self) -> list[str]:
        """List all registered capabilities.

        Returns:
            List of capability names
        """
        return list(self._capability_providers.keys())

    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all capability configurations.

        Returns:
            Dictionary mapping capability names to their configurations
        """
        return {
            name: provider.get_config() for name, provider in self._capability_providers.items()
        }

    def validate_all(self) -> bool:
        """Validate all registered capabilities.

        Returns:
            True if all capabilities are valid
        """
        for name, provider in self._capability_providers.items():
            if not provider.validate():
                self._logger.error(f"Capability validation failed: {name}")
                return False

        return True
