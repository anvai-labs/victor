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

"""Strategy pattern for provider classification and extensibility.

This module provides strategy implementations for provider classification,
following the Open/Closed Principle (OCP). New providers can be added
without modifying existing code.

Part of SOLID-based refactoring to eliminate OCP violations.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Set

from victor.agent.protocols import IProviderClassificationStrategy


class ProviderCategory(str, Enum):
    """Provider deployment category."""

    CLOUD = "cloud"  # Cloud-hosted LLM providers
    LOCAL = "local"  # Locally-hosted LLM providers
    UNKNOWN = "unknown"  # Unclassified providers


class DefaultProviderClassificationStrategy(IProviderClassificationStrategy):
    """Default provider classification strategy.

    Uses hardcoded sets of cloud and local providers. This is the
    default implementation used when no custom strategy is provided.
    """

    def __init__(self) -> None:
        """Initialize with default provider sets."""
        self._cloud_providers: Set[str] = {
            "anthropic",
            "openai",
            "google",
            "xai",
            "deepseek",
            "moonshot",
            "groq",
        }
        self._local_providers: Set[str] = {
            "ollama",
            "lmstudio",
            "vllm",
        }

    def is_cloud_provider(self, provider_name: str) -> bool:
        """Check if provider is cloud-based.

        Args:
            provider_name: Name of the provider

        Returns:
            True if cloud provider, False otherwise
        """
        return provider_name.lower() in self._cloud_providers

    def is_local_provider(self, provider_name: str) -> bool:
        """Check if provider is local.

        Args:
            provider_name: Name of the provider

        Returns:
            True if local provider, False otherwise
        """
        return provider_name.lower() in self._local_providers

    def get_provider_type(self, provider_name: str) -> ProviderCategory:
        """Get provider type category.

        Args:
            provider_name: Name of the provider

        Returns:
            Provider type enum value
        """
        if self.is_cloud_provider(provider_name):
            return ProviderCategory.CLOUD
        if self.is_local_provider(provider_name):
            return ProviderCategory.LOCAL
        return ProviderCategory.UNKNOWN


class ConfigurableProviderClassificationStrategy(IProviderClassificationStrategy):
    """Configurable provider classification strategy.

    Allows adding new providers without modifying code (OCP compliance).
    Supports runtime configuration of provider classifications.
    """

    def __init__(
        self,
        cloud_providers: Set[str] = None,
        local_providers: Set[str] = None,
    ) -> None:
        """Initialize with optional provider sets.

        Args:
            cloud_providers: Set of cloud provider names
            local_providers: Set of local provider names
        """
        self._cloud_providers: Set[str] = cloud_providers or set()
        self._local_providers: Set[str] = local_providers or set()

    def add_cloud_provider(self, provider_name: str) -> None:
        """Add a cloud provider (OCP compliance).

        Args:
            provider_name: Name of the provider to add
        """
        self._cloud_providers.add(provider_name.lower())

    def add_local_provider(self, provider_name: str) -> None:
        """Add a local provider (OCP compliance).

        Args:
            provider_name: Name of the provider to add
        """
        self._local_providers.add(provider_name.lower())

    def remove_cloud_provider(self, provider_name: str) -> None:
        """Remove a cloud provider.

        Args:
            provider_name: Name of the provider to remove
        """
        self._cloud_providers.discard(provider_name.lower())

    def remove_local_provider(self, provider_name: str) -> None:
        """Remove a local provider.

        Args:
            provider_name: Name of the provider to remove
        """
        self._local_providers.discard(provider_name.lower())

    def is_cloud_provider(self, provider_name: str) -> bool:
        """Check if provider is cloud-based.

        Args:
            provider_name: Name of the provider

        Returns:
            True if cloud provider, False otherwise
        """
        return provider_name.lower() in self._cloud_providers

    def is_local_provider(self, provider_name: str) -> bool:
        """Check if provider is local.

        Args:
            provider_name: Name of the provider

        Returns:
            True if local provider, False otherwise
        """
        return provider_name.lower() in self._local_providers

    def get_provider_type(self, provider_name: str) -> ProviderCategory:
        """Get provider type category.

        Args:
            provider_name: Name of the provider

        Returns:
            Provider type enum value
        """
        if self.is_cloud_provider(provider_name):
            return ProviderCategory.CLOUD
        if self.is_local_provider(provider_name):
            return ProviderCategory.LOCAL
        return ProviderCategory.UNKNOWN
