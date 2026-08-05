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

"""Registry for embedding providers."""

import logging
from typing import Dict, List, Type

from victor_coding.codebase.embeddings.base import (
    BaseEmbeddingProvider,
    EmbeddingConfig,
)

logger = logging.getLogger(__name__)


class EmbeddingRegistry:
    """Central registry for embedding providers.

    Similar to the LLM provider registry, this allows plugins to register
    themselves and be discovered/created by name.

    Usage:
        # Register a provider
        EmbeddingRegistry.register("lancedb", LanceDBProvider)

        # Create a provider
        config = EmbeddingConfig(vector_store="lancedb")
        provider = EmbeddingRegistry.create(config)
    """

    _providers: Dict[str, Type[BaseEmbeddingProvider]] = {}

    @classmethod
    def register(cls, name: str, provider_class: Type[BaseEmbeddingProvider]) -> None:
        """Register an embedding provider.

        Args:
            name: Provider name (e.g., "lancedb")
            provider_class: Provider class (must inherit from BaseEmbeddingProvider)
        """
        if not issubclass(provider_class, BaseEmbeddingProvider):
            raise TypeError(f"{provider_class} must inherit from BaseEmbeddingProvider")

        cls._providers[name] = provider_class
        # Route to the logger (stderr/log), never stdout — these import-time
        # registration lines otherwise pollute machine-readable stdout such as
        # `victor auth env`'s `export ...` output.
        logger.debug("Registered embedding provider: %s", name)

    @classmethod
    def get(cls, name: str) -> Type[BaseEmbeddingProvider]:
        """Get a provider class by name.

        Args:
            name: Provider name

        Returns:
            Provider class

        Raises:
            KeyError: If provider not found
        """
        if name not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise KeyError(
                f"Unknown embedding provider: {name}. "
                f"Available: {available if available else 'none'}"
            )

        return cls._providers[name]

    @classmethod
    def create(cls, config: EmbeddingConfig) -> BaseEmbeddingProvider:
        """Create a provider instance from configuration.

        Args:
            config: Embedding configuration

        Returns:
            Initialized provider instance
        """
        provider_class = cls.get(config.vector_store)
        return provider_class(config)

    @classmethod
    def list_providers(cls) -> List[str]:
        """List all registered provider names.

        Returns:
            List of provider names
        """
        return list(cls._providers.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a provider is registered.

        Args:
            name: Provider name

        Returns:
            True if registered, False otherwise
        """
        return name in cls._providers


# Auto-discovery: Import and register all providers
def _auto_register_providers() -> None:
    """Automatically discover and register embedding providers."""
    try:
        from victor_coding.codebase.embeddings.lancedb_provider import LanceDBProvider

        EmbeddingRegistry.register("lancedb", LanceDBProvider)
    except ImportError:
        pass  # LanceDB not installed

    try:
        from victor_coding.codebase.embeddings.proximadb_provider import (
            ProximaDBProvider,
        )

        EmbeddingRegistry.register("proximadb", ProximaDBProvider)
    except ImportError:
        pass  # ProximaDB not installed


# Auto-register on module import
_auto_register_providers()
