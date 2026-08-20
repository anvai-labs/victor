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

"""Base Service Provider for Verticals.

This module provides a reusable service provider base class that verticals
can use for DI container registration. It eliminates the need for each
vertical to implement its own service provider by providing a generic
implementation that works with any vertical's extensions.

Design Philosophy:
- Framework enhancement over vertical-specific code
- All verticals benefit from framework improvements
- Consistent service registration patterns
- Easy to extend for vertical-specific needs

Usage:
    # In a vertical's assistant.py:
    from victor.core.verticals.base_service_provider import BaseVerticalServiceProvider

    class MyVerticalAssistant(VerticalBase):
        @classmethod
        def get_service_provider(cls):
            return BaseVerticalServiceProvider(vertical_name=cls.name)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Type

from victor.core.verticals.protocols import ServiceProviderProtocol

if TYPE_CHECKING:
    from victor.core.container import ServiceContainer
    from victor.core.di import ServiceRegistrationSpec
    from victor.core.verticals.protocols import (
        ModeConfigProviderProtocol,
        ToolDependencyProviderProtocol,
        VerticalPromptProtocol,
        VerticalSafetyProtocol,
    )
    from victor.config.settings import Settings
    from victor.core.di import register_services_from_specs

logger = logging.getLogger(__name__)


# Protocol definitions for vertical services
class VerticalMiddlewareProtocol:
    """Protocol for vertical middleware."""

    pass


class VerticalSafetyProtocol:
    """Protocol for vertical safety extension."""

    pass


class VerticalPromptProtocol:
    """Protocol for vertical prompt contributor."""

    pass


class BaseVerticalServiceProvider(ServiceProviderProtocol):
    """Generic service provider for any vertical.

    Provides DI container registration for common vertical services:
    - Safety extensions
    - Prompt contributors
    - Mode config providers
    - Tool dependency providers

    This base class can be used by any vertical without customization,
    or extended for vertical-specific service registration.

    Example:
        # Simple usage (automatic from VerticalBase):
        class ResearchAssistant(VerticalBase):
            pass  # Automatically gets BaseVerticalServiceProvider

        # Custom usage:
        class ResearchAssistant(VerticalBase):
            @classmethod
            def get_service_provider(cls):
                provider = BaseVerticalServiceProvider(vertical_name="research")
                return provider

        # Extended usage:
        class ResearchServiceProvider(BaseVerticalServiceProvider):
            def register_services(self, container, settings):
                super().register_services(container, settings)
                # Add research-specific services
                self._register_custom_services(container, settings)

    Attributes:
        vertical_name: Name of the vertical (for logging)
    """

    def __init__(self, vertical_name: str = "unknown"):
        """Initialize the service provider.

        Args:
            vertical_name: Name of the vertical for logging purposes
        """
        self.vertical_name = vertical_name

    def register_services(
        self,
        container: "ServiceContainer",
        settings: "Settings",
    ) -> None:
        """Register vertical-specific services with DI container.

        Registers common services that most verticals need:
        - Mode config provider (if available)
        - Tool dependency provider (if available)
        - Prompt contributor (if available)
        - Safety extension (if available)

        Args:
            container: DI container to register services in
            settings: Application settings
        """
        specs: List[ServiceRegistrationSpec] = []

        mode_config_provider = self._get_mode_config_provider()
        if mode_config_provider is not None:
            specs.append(
                ServiceRegistrationSpec(
                    protocol=ModeConfigProviderProtocol,
                    factory=lambda _: mode_config_provider,
                )
            )

        tool_dep_provider = self._get_tool_dependency_provider()
        if tool_dep_provider is not None:
            specs.append(
                ServiceRegistrationSpec(
                    protocol=ToolDependencyProviderProtocol,
                    factory=lambda _: tool_dep_provider,
                )
            )

        prompt_contributor = self._get_prompt_contributor()
        if prompt_contributor is not None:
            specs.append(
                ServiceRegistrationSpec(
                    protocol=VerticalPromptProtocol,
                    factory=lambda _: prompt_contributor,
                )
            )

        safety_extension = self._get_safety_extension()
        if safety_extension is not None:
            specs.append(
                ServiceRegistrationSpec(
                    protocol=VerticalSafetyProtocol,
                    factory=lambda _: safety_extension,
                )
            )

        register_services_from_specs(container, specs)

        logger.info("Registered %s vertical services", self.vertical_name)

    def _register_mode_config(
        self,
        container: "ServiceContainer",
        settings: "Settings",
    ) -> None:
        """Register mode configuration provider if available."""
        from victor.core.container import ServiceLifetime
        from victor.core.verticals.protocols import ModeConfigProviderProtocol

        # Get the vertical class to retrieve mode config provider
        mode_config_provider = self._get_mode_config_provider()
        if mode_config_provider is None:
            return

        def create_mode_config(_):
            return mode_config_provider

        container.register(
            ModeConfigProviderProtocol,
            create_mode_config,
            ServiceLifetime.SINGLETON,
        )
        logger.debug("Registered %s mode config provider", self.vertical_name)

    def _register_tool_dependencies(
        self,
        container: "ServiceContainer",
        settings: "Settings",
    ) -> None:
        """Register tool dependency provider if available."""
        from victor.core.container import ServiceLifetime
        from victor.core.verticals.protocols import ToolDependencyProviderProtocol

        tool_dep_provider = self._get_tool_dependency_provider()
        if tool_dep_provider is None:
            return

        def create_tool_deps(_):
            return tool_dep_provider

        container.register(
            ToolDependencyProviderProtocol,
            create_tool_deps,
            ServiceLifetime.SINGLETON,
        )
        logger.debug("Registered %s tool dependency provider", self.vertical_name)

    def _register_prompts(
        self,
        container: "ServiceContainer",
        settings: "Settings",
    ) -> None:
        """Register prompt contributor if available."""
        from victor.core.container import ServiceLifetime

        prompt_contributor = self._get_prompt_contributor()
        if prompt_contributor is None:
            return

        def create_prompts(_):
            return prompt_contributor

        container.register(
            VerticalPromptProtocol,
            create_prompts,
            ServiceLifetime.SINGLETON,
        )
        logger.debug("Registered %s prompt contributor", self.vertical_name)

    def _register_safety(
        self,
        container: "ServiceContainer",
        settings: "Settings",
    ) -> None:
        """Register safety extension if available."""
        from victor.core.container import ServiceLifetime

        safety_extension = self._get_safety_extension()
        if safety_extension is None:
            return

        def create_safety(_):
            return safety_extension

        container.register(
            VerticalSafetyProtocol,
            create_safety,
            ServiceLifetime.SINGLETON,
        )
        logger.debug("Registered %s safety extension", self.vertical_name)

    def _get_mode_config_provider(self) -> Any:
        """Get mode config provider from the vertical.

        Override in subclass or set via constructor for custom provider.

        Returns:
            Mode config provider instance or None
        """
        return None

    def _get_tool_dependency_provider(self) -> Any:
        """Get tool dependency provider from the vertical.

        Override in subclass or set via constructor for custom provider.

        Returns:
            Tool dependency provider instance or None
        """
        return None

    def _get_prompt_contributor(self) -> Any:
        """Get prompt contributor from the vertical.

        Override in subclass or set via constructor for custom provider.

        Returns:
            Prompt contributor instance or None
        """
        return None

    def _get_safety_extension(self) -> Any:
        """Get safety extension from the vertical.

        Override in subclass or set via constructor for custom provider.

        Returns:
            Safety extension instance or None
        """
        return None

    def get_required_services(self) -> List[Type]:
        """Get list of required service types.

        Returns:
            Empty list - no hard requirements by default
        """
        return []

    def get_optional_services(self) -> List[Type]:
        """Get list of optional service types.

        Returns:
            List of optional protocol types
        """
        from victor.core.verticals.protocols import (
            ModeConfigProviderProtocol,
            ToolDependencyProviderProtocol,
        )

        return [
            VerticalPromptProtocol,
            VerticalSafetyProtocol,
            ModeConfigProviderProtocol,
            ToolDependencyProviderProtocol,
        ]


class VerticalServiceProviderFactory:
    """Factory for creating vertical-specific service providers.

    Creates service providers that are pre-configured with vertical
    extensions, making it easy to get full DI support for any vertical.

    Usage:
        # Create a service provider for a vertical class
        provider = VerticalServiceProviderFactory.create(ResearchAssistant)

        # Register services
        provider.register_services(container, settings)
    """

    @staticmethod
    def create(vertical_class: Type) -> BaseVerticalServiceProvider:
        """Create a service provider for a vertical.

        The created provider is pre-configured with the vertical's
        extensions (prompt contributor, safety extension, etc.).

        Args:
            vertical_class: Vertical class to create provider for

        Returns:
            Configured BaseVerticalServiceProvider
        """
        provider = _ConfiguredVerticalServiceProvider(
            vertical_name=getattr(vertical_class, "name", "unknown"),
            vertical_class=vertical_class,
        )
        return provider


class _ConfiguredVerticalServiceProvider(BaseVerticalServiceProvider):
    """Service provider configured with a specific vertical's extensions.

    Internal class used by VerticalServiceProviderFactory.
    """

    def __init__(self, vertical_name: str, vertical_class: Type):
        """Initialize with vertical class.

        Args:
            vertical_name: Name of the vertical
            vertical_class: Vertical class to get extensions from
        """
        super().__init__(vertical_name)
        self._vertical_class = vertical_class

    def _get_mode_config_provider(self) -> Any:
        """Get mode config provider from vertical class."""
        if hasattr(self._vertical_class, "get_mode_config_provider"):
            return self._vertical_class.get_mode_config_provider()
        return None

    def _get_tool_dependency_provider(self) -> Any:
        """Get tool dependency provider from vertical class."""
        if hasattr(self._vertical_class, "get_tool_dependency_provider"):
            return self._vertical_class.get_tool_dependency_provider()
        return None

    def _get_prompt_contributor(self) -> Any:
        """Get prompt contributor from vertical class."""
        if hasattr(self._vertical_class, "get_prompt_contributor"):
            return self._vertical_class.get_prompt_contributor()
        return None

    def _get_safety_extension(self) -> Any:
        """Get safety extension from vertical class."""
        if hasattr(self._vertical_class, "get_safety_extension"):
            return self._vertical_class.get_safety_extension()
        return None


__all__ = [
    "BaseVerticalServiceProvider",
    "VerticalServiceProviderFactory",
    "VerticalMiddlewareProtocol",
    "VerticalSafetyProtocol",
    "VerticalPromptProtocol",
]
