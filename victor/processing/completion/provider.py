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

"""Completion provider interface and base implementation.

Defines the abstract interface for completion providers following
the Strategy pattern for extensibility.
"""

import logging
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional, Protocol, runtime_checkable

from victor.processing.completion.protocol import (
    CompletionCapabilities,
    CompletionItem,
    CompletionList,
    CompletionParams,
    InlineCompletionList,
    InlineCompletionParams,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class CompletionProvider(Protocol):
    """Protocol for completion providers.

    Completion providers supply code completions from various sources:
    - LSP servers (language-aware completions)
    - AI models (intelligent suggestions)
    - Snippets (template-based completions)
    - Custom sources (project-specific)
    """

    @property
    def name(self) -> str:
        """Unique identifier for this provider."""
        ...

    @property
    def priority(self) -> int:
        """Provider priority (higher = checked first)."""
        ...

    def get_capabilities(self) -> CompletionCapabilities:
        """Return the capabilities of this provider."""
        ...

    async def provide_completions(self, params: CompletionParams) -> CompletionList:
        """Provide completion items for the given parameters.

        Args:
            params: Completion request parameters including file, position, context

        Returns:
            CompletionList with completion items
        """
        ...

    async def provide_inline_completions(
        self, params: InlineCompletionParams
    ) -> InlineCompletionList:
        """Provide inline (ghost text) completions.

        Args:
            params: Inline completion parameters

        Returns:
            InlineCompletionList with inline suggestions
        """
        ...

    async def resolve_completion(self, item: CompletionItem) -> CompletionItem:
        """Resolve additional details for a completion item.

        Args:
            item: The item to resolve

        Returns:
            The item with additional details filled in
        """
        ...


class BaseCompletionProvider(ABC):
    """Abstract base class for completion providers.

    Provides common functionality and default implementations.
    Subclasses must implement the abstract methods.
    """

    def __init__(self, priority: int = 50):
        """Initialize the provider.

        Args:
            priority: Provider priority (default 50, range 0-100)
        """
        self._priority = priority
        self._enabled = True

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this provider."""
        ...

    @property
    def priority(self) -> int:
        """Provider priority (higher = checked first)."""
        return self._priority

    @priority.setter
    def priority(self, value: int) -> None:
        """Set provider priority."""
        self._priority = max(0, min(100, value))

    @property
    def enabled(self) -> bool:
        """Whether this provider is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable this provider."""
        self._enabled = value

    @abstractmethod
    def get_capabilities(self) -> CompletionCapabilities:
        """Return the capabilities of this provider."""
        ...

    @abstractmethod
    async def provide_completions(self, params: CompletionParams) -> CompletionList:
        """Provide completion items for the given parameters."""
        ...

    async def provide_inline_completions(
        self, params: InlineCompletionParams
    ) -> InlineCompletionList:
        """Provide inline completions.

        Default implementation returns empty list.
        Override in subclasses that support inline completions.
        """
        return InlineCompletionList(items=[])

    async def resolve_completion(self, item: CompletionItem) -> CompletionItem:
        """Resolve additional details for a completion item.

        Default implementation returns the item unchanged.
        Override in subclasses that support resolution.
        """
        return item

    def supports_language(self, language: str) -> bool:
        """Check if this provider supports a language.

        Args:
            language: Language identifier (e.g., 'python', 'javascript')

        Returns:
            True if supported, False otherwise
        """
        capabilities = self.get_capabilities()
        if not capabilities.supported_languages:
            return True  # Empty means all languages
        return language.lower() in [lang.lower() for lang in capabilities.supported_languages]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, priority={self.priority})"


class StreamingCompletionProvider(BaseCompletionProvider):
    """Base class for providers that support streaming completions."""

    @abstractmethod
    async def stream_inline_completion(self, params: InlineCompletionParams) -> AsyncIterator[str]:
        """Stream inline completion tokens.

        Args:
            params: Inline completion parameters

        Yields:
            Completion text tokens as they're generated
        """
        ...

    def get_capabilities(self) -> CompletionCapabilities:
        """Return capabilities with streaming enabled."""
        caps = self._get_base_capabilities()
        caps.supports_streaming = True
        return caps

    @abstractmethod
    def _get_base_capabilities(self) -> CompletionCapabilities:
        """Get base capabilities without streaming flag."""
        ...


class CachingCompletionProvider(BaseCompletionProvider):
    """Base class for providers with completion caching."""

    def __init__(self, priority: int = 50, cache_ttl: int = 300):
        """Initialize with caching.

        Args:
            priority: Provider priority
            cache_ttl: Cache time-to-live in seconds
        """
        super().__init__(priority)
        self._cache_ttl = cache_ttl
        self._cache: dict[str, tuple[float, CompletionList]] = {}

    def _cache_key(self, params: CompletionParams) -> str:
        """Generate cache key for completion params."""
        return (
            f"{params.file_path}:{params.position.line}:{params.position.character}:{params.prefix}"
        )

    def _get_cached(self, params: CompletionParams) -> Optional[CompletionList]:
        """Get cached completions if available and not expired."""
        import time

        key = self._cache_key(params)
        if key in self._cache:
            timestamp, result = self._cache[key]
            if time.time() - timestamp < self._cache_ttl:
                return result
            del self._cache[key]
        return None

    def _set_cached(self, params: CompletionParams, result: CompletionList) -> None:
        """Cache completion results."""
        import time

        key = self._cache_key(params)
        self._cache[key] = (time.time(), result)

    def clear_cache(self) -> None:
        """Clear all cached completions."""
        self._cache.clear()


class CompositeCompletionProvider(BaseCompletionProvider):
    """Provider that combines results from multiple providers.

    Implements the Composite pattern for aggregating completions.
    """

    def __init__(self, providers: Optional[list[BaseCompletionProvider]] = None):
        """Initialize with child providers.

        Args:
            providers: List of providers to aggregate
        """
        super().__init__(priority=100)  # Highest priority
        self._providers = providers or []

    @property
    def name(self) -> str:
        return "composite"

    def add_provider(self, provider: BaseCompletionProvider) -> None:
        """Add a provider to the composite."""
        self._providers.append(provider)
        self._providers.sort(key=lambda p: p.priority, reverse=True)

    def remove_provider(self, name: str) -> bool:
        """Remove a provider by name."""
        for i, provider in enumerate(self._providers):
            if provider.name == name:
                del self._providers[i]
                return True
        return False

    def get_capabilities(self) -> CompletionCapabilities:
        """Aggregate capabilities from all providers."""
        caps = CompletionCapabilities()
        trigger_chars: set[str] = set()
        languages: set[str] = set()

        for provider in self._providers:
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

    async def provide_completions(self, params: CompletionParams) -> CompletionList:
        """Aggregate completions from all providers."""
        all_items: list[CompletionItem] = []
        is_incomplete = False

        for provider in self._providers:
            if not provider.enabled:
                continue
            if not provider.supports_language(params.language):
                continue

            try:
                result = await provider.provide_completions(params)
                all_items.extend(result.items)
                is_incomplete |= result.is_incomplete
            except Exception as e:
                logger.warning(f"Provider {provider.name} failed: {e}")
                continue

        # Sort by confidence, then by sort_text
        all_items.sort(key=lambda item: (-item.confidence, item.sort_text or item.label))

        # Limit results
        if len(all_items) > params.max_results:
            all_items = all_items[: params.max_results]
            is_incomplete = True

        return CompletionList(is_incomplete=is_incomplete, items=all_items)

    async def provide_inline_completions(
        self, params: InlineCompletionParams
    ) -> InlineCompletionList:
        """Get inline completions from first capable provider."""
        for provider in self._providers:
            if not provider.enabled:
                continue
            if not provider.supports_language(params.language):
                continue

            caps = provider.get_capabilities()
            if not caps.supports_inline_completion:
                continue

            try:
                result = await provider.provide_inline_completions(params)
                if result.items:
                    return result
            except Exception as e:
                logger.warning(f"Provider {provider.name} failed inline completion: {e}")
                continue

        return InlineCompletionList(items=[])
