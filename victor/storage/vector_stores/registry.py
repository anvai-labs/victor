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

import hashlib
import logging
import threading
from typing import Dict, List, Optional, Type

from victor.storage.vector_stores.base import BaseEmbeddingProvider, EmbeddingConfig

logger = logging.getLogger(__name__)


class EmbeddingRegistry:
    """Central registry for embedding providers.

    Similar to the LLM provider registry, this allows plugins to register
    themselves and be discovered/created by name.

    **Singleton Pattern**: Providers are cached per unique configuration to
    prevent concurrent access issues with file-based databases like LanceDB.
    Multiple orchestrators/agents will share the same provider instance for
    a given persist_directory.

    Usage:
        # Register a provider
        EmbeddingRegistry.register("lancedb", LanceDBProvider)

        # Create a provider (cached - returns same instance for same config)
        config = EmbeddingConfig(vector_store="lancedb", persist_directory="~/.victor/embeddings")
        provider = EmbeddingRegistry.create(config)

        # Clear cache (mainly for testing)
        EmbeddingRegistry.reset()
    """

    _providers: Dict[str, Type[BaseEmbeddingProvider]] = {}

    # Singleton cache: {config_key: provider_instance}
    _provider_cache: Dict[str, BaseEmbeddingProvider] = {}
    _cache_lock = threading.Lock()

    @classmethod
    def register(cls, name: str, provider_class: Type[BaseEmbeddingProvider]) -> None:
        """Register an embedding provider.

        Args:
            name: Provider name (e.g., "lancedb", "proximadb")
            provider_class: Provider class (must inherit from BaseEmbeddingProvider)
        """
        if not issubclass(provider_class, BaseEmbeddingProvider):
            raise TypeError(f"{provider_class} must inherit from BaseEmbeddingProvider")

        cls._providers[name] = provider_class
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

            # Special case for victor-coding bridge provider
            if name == "victor_structural_bridge":
                # Check if victor-coding is actually installed
                import importlib.util

                victor_coding_installed = importlib.util.find_spec("victor_coding") is not None

                if victor_coding_installed:
                    # victor-coding IS installed but the bridge isn't registered in victor's registry
                    # This is expected - the bridge lives in victor-coding's namespace
                    # Provide a helpful error message that explains this
                    raise KeyError(
                        f"The '{name}' vector store is only available through victor-coding's "
                        f"embedding registry, not victor's core registry. "
                        f"Use the upstream vector store directly (e.g., 'lancedb') "
                        f"or access through victor-coding's CodebaseIndex factory.\n"
                        f"Available providers in victor's registry: {available if available else 'none'}"
                    )
                else:
                    raise KeyError(
                        f"The '{name}' vector store requires victor-coding to be installed. "
                        f"Install it with: pip install victor-coding\n"
                        f"Available providers in this registry: {available if available else 'none'}"
                    )

            raise KeyError(
                f"Unknown embedding provider: {name}. "
                f"Available: {available if available else 'none'}"
            )

        return cls._providers[name]

    @classmethod
    def _get_config_key(cls, config: EmbeddingConfig) -> str:
        """Generate a unique cache key for the configuration.

        Key components:
        - vector_store: The provider type (lancedb, etc.)
        - persist_directory: The database path (critical for singleton)
        - embedding_model_type: Model type (sentence-transformers, ollama, etc.)
        - embedding_model_name: Model name (affects embedding dimension)

        Args:
            config: Embedding configuration

        Returns:
            SHA256 hash of the configuration key components
        """
        # Normalize the persist directory path
        persist_dir = config.persist_directory or "default"

        # Create key from critical configuration elements
        key_parts = [
            config.vector_store,
            persist_dir,
            config.embedding_model_type,
            config.embedding_model_name,
            config.distance_metric,
        ]

        # Join and hash
        key_str = "|".join(str(part) for part in key_parts)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    @classmethod
    def create(cls, config: EmbeddingConfig) -> BaseEmbeddingProvider:
        """Create or retrieve a cached provider instance from configuration.

        **Singleton Pattern**: Returns the same provider instance for identical
        configurations to prevent concurrent access issues with file-based databases.

        Args:
            config: Embedding configuration

        Returns:
            Initialized provider instance (cached if exists)
        """
        # Generate cache key from configuration
        cache_key = cls._get_config_key(config)

        # Check cache first (fast path, no lock)
        if cache_key in cls._provider_cache:
            logger.debug(f"[EmbeddingRegistry] Cache hit for key: {cache_key}")
            return cls._provider_cache[cache_key]

        # Cache miss - acquire lock and create instance
        with cls._cache_lock:
            # Double-checked locking pattern
            if cache_key in cls._provider_cache:
                return cls._provider_cache[cache_key]

            # Create new provider instance
            provider_class = cls.get(config.vector_store)
            provider = provider_class(config)

            # Cache the instance
            cls._provider_cache[cache_key] = provider
            logger.info(
                f"[EmbeddingRegistry] Created and cached {config.vector_store} provider "
                f"(key: {cache_key}, cache size: {len(cls._provider_cache)})"
            )

            return provider

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

    @classmethod
    def reset(cls) -> None:
        """Reset the provider cache (mainly for testing)."""
        with cls._cache_lock:
            if cls._provider_cache:
                logger.warning(
                    "[EmbeddingRegistry] reset() called with %d active providers. "
                    "Call close_all() first to properly release resources.",
                    len(cls._provider_cache),
                )
            cls._provider_cache.clear()
            logger.info("[EmbeddingRegistry] Provider cache cleared")

    @classmethod
    async def close_all(cls) -> None:
        """Close all cached providers and clear the cache.

        Call this before reset() or application shutdown to properly
        release database connections and embedding model resources.
        """
        with cls._cache_lock:
            for key, provider in list(cls._provider_cache.items()):
                try:
                    await provider.close()
                except Exception:
                    logger.warning("Error closing provider %s", key, exc_info=True)
            cls._provider_cache.clear()
            logger.info("[EmbeddingRegistry] All providers closed and cache cleared")

    @classmethod
    def get_cache_stats(cls) -> Dict[str, int]:
        """Get statistics about the provider cache.

        Returns:
            Dictionary with cache size and provider breakdown
        """
        with cls._cache_lock:
            provider_types = {}
            for provider in cls._provider_cache.values():
                provider_type = provider.__class__.__name__
                provider_types[provider_type] = provider_types.get(provider_type, 0) + 1

            return {
                "total_cached_providers": len(cls._provider_cache),
                "provider_types": provider_types,
            }


# Auto-discovery: Import and register all providers
def _auto_register_providers() -> None:
    """Automatically discover and register embedding providers."""
    try:
        from victor.storage.vector_stores.lancedb_provider import LanceDBProvider

        EmbeddingRegistry.register("lancedb", LanceDBProvider)
    except ImportError:
        pass  # LanceDB not installed

    try:
        from victor.storage.vector_stores.proximadb_provider import ProximaDBProvider

        EmbeddingRegistry.register("proximadb", ProximaDBProvider)
    except ImportError:
        pass  # ProximaDB not installed

    try:
        from victor.storage.vector_stores.proximadb_multi import (
            ProximaDBMultiModelProvider,
        )

        EmbeddingRegistry.register("proximadb_multi", ProximaDBMultiModelProvider)
    except Exception:
        pass  # ProximaDB SDK not installed


# Auto-register on module import
_auto_register_providers()
