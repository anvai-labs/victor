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

"""Tool adapter coordination.

This module provides ToolAdapterCoordinator, which handles tool calling
adapter initialization and management. Extracted from ProviderManager to
follow the Single Responsibility Principle (SRP).

Part of SOLID-based refactoring to eliminate god class anti-pattern.
"""

import logging
from typing import Any, Optional

from victor.agent.protocols import IToolAdapterCoordinator, IProviderSwitcher
from victor.agent.tool_calling import (
    ToolCallingAdapterRegistry,
    ToolCallingCapabilities,
)

logger = logging.getLogger(__name__)


class ToolAdapterCoordinator(IToolAdapterCoordinator):
    """Coordinates tool adapter initialization and capabilities.

    This class is responsible for:
    - Initializing tool calling adapters for providers
    - Getting tool calling capabilities
    - Managing adapter lifecycle
    - Coordinating with provider switcher

    SRP Compliance: Focuses only on tool adapter coordination, delegating
    provider switching and health monitoring to specialized components.

    Attributes:
        _provider_switcher: Provider switcher for current provider info
        _adapter: Current tool calling adapter
        _capabilities: Current tool calling capabilities
        _settings: Application settings for adapter configuration
    """

    def __init__(
        self,
        provider_switcher: IProviderSwitcher,
        settings: Optional[Any] = None,
    ):
        """Initialize the tool adapter coordinator.

        Args:
            provider_switcher: Provider switcher for current provider info
            settings: Optional application settings
        """
        self._provider_switcher = provider_switcher
        self._settings = settings
        self._adapter: Optional[Any] = None
        self._capabilities: Optional[ToolCallingCapabilities] = None

    def initialize_adapter(self) -> ToolCallingCapabilities:
        """Initialize tool calling adapter for current provider/model.

        Returns:
            Tool calling capabilities

        Raises:
            ValueError: If no provider is configured
        """
        provider_name = self._provider_switcher.get_current_state()
        if not provider_name:
            raise ValueError("No provider configured")

        _provider = self._provider_switcher.get_current_provider()
        model = self._provider_switcher.get_current_model()

        # Get adapter from registry
        adapter = ToolCallingAdapterRegistry.get_adapter(
            provider_name=provider_name.provider_name,
            model=model,
            config={"settings": self._settings} if self._settings else {},
        )

        capabilities = adapter.get_capabilities()

        self._adapter = adapter
        self._capabilities = capabilities

        logger.debug(
            f"Tool adapter initialized for {provider_name.provider_name}:{model}: "
            f"native={capabilities.native_tool_calls}, "
            f"format={capabilities.tool_call_format.value}"
        )

        return capabilities

    def get_capabilities(self) -> ToolCallingCapabilities:
        """Get tool calling capabilities.

        Returns:
            Tool calling capabilities

        Raises:
            ValueError: If adapter not initialized
        """
        if self._capabilities is None:
            raise ValueError("Tool adapter not initialized. Call initialize_adapter() first.")

        return self._capabilities

    def get_adapter(self) -> Any:
        """Get current tool adapter instance.

        Returns:
            Tool adapter instance

        Raises:
            ValueError: If adapter not initialized
        """
        if self._adapter is None:
            raise ValueError("Tool adapter not initialized. Call initialize_adapter() first.")

        return self._adapter

    def is_initialized(self) -> bool:
        """Check if adapter has been initialized.

        Returns:
            True if adapter is initialized, False otherwise
        """
        return self._adapter is not None
