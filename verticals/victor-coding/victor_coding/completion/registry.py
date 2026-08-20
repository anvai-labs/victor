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

"""Completion provider registry with auto-discovery.

Manages registration and lookup of completion providers following
the Factory pattern.
"""

import importlib
import logging
from pathlib import Path
from typing import Callable, Optional, Type

from victor_coding.completion.protocol import CompletionCapabilities
from victor_coding.completion.provider import BaseCompletionProvider

logger = logging.getLogger(__name__)


class CompletionProviderRegistry:
    """Registry for completion providers.

    Supports:
    - Manual registration of providers
    - Auto-discovery of built-in providers
    - Factory-based lazy instantiation
    - Priority-based provider ordering
    """

    def __init__(self):
        """Initialize the registry."""
        self._providers: dict[str, BaseCompletionProvider] = {}
        self._factories: dict[str, Callable[[], BaseCompletionProvider]] = {}
        self._provider_classes: dict[str, Type[BaseCompletionProvider]] = {}

    def register(self, provider: BaseCompletionProvider) -> None:
        """Register a provider instance.

        Args:
            provider: The provider to register
        """
        if provider.name in self._providers:
            logger.warning(f"Overwriting existing provider: {provider.name}")
        self._providers[provider.name] = provider
        logger.debug(f"Registered completion provider: {provider.name}")

    def register_class(
        self,
        name: str,
        provider_class: Type[BaseCompletionProvider],
    ) -> None:
        """Register a provider class for lazy instantiation.

        Args:
            name: Provider name
            provider_class: The provider class
        """
        self._provider_classes[name] = provider_class
        logger.debug(f"Registered provider class: {name}")

    def register_factory(
        self,
        name: str,
        factory: Callable[[], BaseCompletionProvider],
    ) -> None:
        """Register a factory function for lazy instantiation.

        Args:
            name: Provider name
            factory: Factory function that creates the provider
        """
        self._factories[name] = factory
        logger.debug(f"Registered provider factory: {name}")

    def get_provider(self, name: str) -> Optional[BaseCompletionProvider]:
        """Get a provider by name.

        Tries in order:
        1. Already instantiated providers
        2. Factory functions
        3. Registered classes

        Args:
            name: Provider name

        Returns:
            The provider instance or None if not found
        """
        # Check instantiated providers
        if name in self._providers:
            return self._providers[name]

        # Try factory
        if name in self._factories:
            try:
                provider = self._factories[name]()
                self._providers[name] = provider
                return provider
            except Exception as e:
                logger.error(f"Factory failed for {name}: {e}")
                return None

        # Try class
        if name in self._provider_classes:
            try:
                provider = self._provider_classes[name]()
                self._providers[name] = provider
                return provider
            except Exception as e:
                logger.error(f"Class instantiation failed for {name}: {e}")
                return None

        return None

    def get_all_providers(self) -> list[BaseCompletionProvider]:
        """Get all registered providers, sorted by priority.

        Returns:
            List of providers sorted by priority (highest first)
        """
        # Ensure all factories/classes are instantiated
        all_names = (
            set(self._providers.keys())
            | set(self._factories.keys())
            | set(self._provider_classes.keys())
        )

        for name in all_names:
            self.get_provider(name)

        # Return sorted by priority
        return sorted(
            self._providers.values(),
            key=lambda p: p.priority,
            reverse=True,
        )

    def get_providers_for_language(self, language: str) -> list[BaseCompletionProvider]:
        """Get providers that support a specific language.

        Args:
            language: Language identifier

        Returns:
            List of supporting providers sorted by priority
        """
        return [p for p in self.get_all_providers() if p.enabled and p.supports_language(language)]

    def get_providers_with_capability(self, capability: str) -> list[BaseCompletionProvider]:
        """Get providers with a specific capability.

        Args:
            capability: Capability name (e.g., 'supports_inline_completion')

        Returns:
            List of providers with the capability
        """
        result = []
        for provider in self.get_all_providers():
            if not provider.enabled:
                continue
            caps = provider.get_capabilities()
            if getattr(caps, capability, False):
                result.append(provider)
        return result

    def unregister(self, name: str) -> bool:
        """Unregister a provider.

        Args:
            name: Provider name

        Returns:
            True if provider was found and removed
        """
        found = False
        if name in self._providers:
            del self._providers[name]
            found = True
        if name in self._factories:
            del self._factories[name]
            found = True
        if name in self._provider_classes:
            del self._provider_classes[name]
            found = True
        return found

    def list_providers(self) -> list[str]:
        """List all registered provider names.

        Returns:
            List of provider names
        """
        return list(
            set(self._providers.keys())
            | set(self._factories.keys())
            | set(self._provider_classes.keys())
        )

    def get_aggregated_capabilities(self) -> CompletionCapabilities:
        """Get combined capabilities from all providers.

        Returns:
            Aggregated capabilities
        """
        caps = CompletionCapabilities()
        trigger_chars: set[str] = set()
        languages: set[str] = set()

        for provider in self.get_all_providers():
            if not provider.enabled:
                continue
            pcaps = provider.get_capabilities()
            caps.supports_completion |= pcaps.supports_completion
            caps.supports_inline_completion |= pcaps.supports_inline_completion
            caps.supports_resolve |= pcaps.supports_resolve
            caps.supports_snippets |= pcaps.supports_snippets
            caps.supports_multi_line |= pcaps.supports_multi_line
            caps.supports_streaming |= pcaps.supports_streaming
            trigger_chars.update(pcaps.trigger_characters)
            languages.update(pcaps.supported_languages)

        caps.trigger_characters = sorted(trigger_chars)
        caps.supported_languages = sorted(languages)
        return caps

    def discover_builtin_providers(self) -> None:
        """Auto-discover and register built-in providers."""
        providers_dir = Path(__file__).parent / "providers"
        if not providers_dir.exists():
            return

        for path in providers_dir.glob("*.py"):
            if path.name.startswith("_"):
                continue

            module_name = f"victor.processing.completion.providers.{path.stem}"
            try:
                module = importlib.import_module(module_name)  # nosemgrep

                # Look for provider classes
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, BaseCompletionProvider)
                        and attr is not BaseCompletionProvider
                        and not attr_name.startswith("_")
                    ):
                        # Check if it has a no-arg constructor
                        try:
                            provider = attr()
                            self.register(provider)
                        except TypeError:
                            # Needs arguments, register as class
                            self.register_class(
                                attr_name.lower().replace("provider", ""),
                                attr,
                            )
            except Exception as e:
                logger.debug(f"Could not load provider module {module_name}: {e}")

    def clear(self) -> None:
        """Clear all registered providers."""
        self._providers.clear()
        self._factories.clear()
        self._provider_classes.clear()


# Global registry singleton
_completion_registry: Optional[CompletionProviderRegistry] = None


def get_completion_registry() -> CompletionProviderRegistry:
    """Get the global completion provider registry.

    Returns:
        The singleton registry instance
    """
    global _completion_registry
    if _completion_registry is None:
        _completion_registry = CompletionProviderRegistry()
        _completion_registry.discover_builtin_providers()
    return _completion_registry


def reset_completion_registry() -> None:
    """Reset the global completion registry.

    Useful for testing.
    """
    global _completion_registry
    _completion_registry = None
