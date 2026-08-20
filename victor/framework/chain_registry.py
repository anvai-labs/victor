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

"""Registry for composed chains across verticals.

Enables discovery and access to Runnable chains from any vertical.

Design Philosophy:
- Singleton pattern for global chain registry
- Thread-safe operations for concurrent access
- Namespace support for vertical-specific chains
- Metadata support for discoverability
- Factory pattern for deferred chain creation
- Decorator-based declarative registration

Usage:
    from victor.framework.chain_registry import (
        ChainRegistry,
        get_chain_registry,
        register_chain,
        get_chain,
        chain,  # decorator
    )

    # Get the global registry
    registry = get_chain_registry()

    # Register a chain (direct)
    registry.register(
        "code_review",
        code_review_chain,
        vertical="coding",
        description="Chain for reviewing code changes",
    )

    # Register a chain factory (deferred creation)
    registry.register_factory(
        "read_analyze",
        lambda: read_tool | analyze_tool,
        vertical="coding",
    )

    # Register using decorator (preferred)
    @chain("coding:read_analyze", description="Read and analyze code")
    def read_and_analyze():
        return as_runnable(read) | code_search

    # Get a chain
    chain = registry.get("code_review", vertical="coding")

    # Create from factory
    runnable = registry.create("coding:read_analyze")

    # List available chains
    chains = registry.list_chains()
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, Generator, List, Optional, Set, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)

# Singleton instance
_registry_instance: Optional["ChainRegistry"] = None
_registry_lock = threading.Lock()


@dataclass
class ChainMetadata:
    """Metadata for a registered chain.

    Attributes:
        name: Short name of the chain
        vertical: Optional vertical namespace (e.g., "coding", "devops")
        description: Human-readable description
        input_type: Optional type hint for chain input
        output_type: Optional type hint for chain output
        tags: Tags for filtering/discovery
        is_factory: Whether this is a factory (deferred creation) vs direct chain
        version: Optional version string for the chain
    """

    name: str
    vertical: Optional[str] = None
    description: str = ""
    input_type: Optional[str] = None
    output_type: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    is_factory: bool = False
    version: str = "1.0.0"

    @property
    def full_name(self) -> str:
        """Get the full qualified name with namespace."""
        if self.vertical:
            return f"{self.vertical}:{self.name}"
        return self.name

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata to dictionary."""
        return {
            "name": self.name,
            "vertical": self.vertical,
            "full_name": self.full_name,
            "description": self.description,
            "input_type": self.input_type,
            "output_type": self.output_type,
            "tags": self.tags,
            "is_factory": self.is_factory,
            "version": self.version,
        }


class ChainRegistry:
    """Registry for composed chains across verticals.

    Chains are namespaced by vertical: "{vertical}:{name}"

    This class provides a centralized location for registering and
    querying Runnable chains from any vertical.

    Thread-safe for concurrent access.

    Supports two registration modes:
    1. Direct registration: Register an already-created chain object
    2. Factory registration: Register a factory function for deferred creation

    Example:
        registry = ChainRegistry.get_instance()

        # Register a chain (direct)
        registry.register(
            "code_review",
            code_review_chain,
            vertical="coding",
            description="Chain for reviewing code",
        )

        # Register a factory (deferred creation)
        registry.register_factory(
            "read_analyze",
            lambda: read_tool | analyze_tool,
            vertical="coding",
        )

        # Get direct chain
        chain = registry.get("code_review", vertical="coding")

        # Create from factory
        runnable = registry.create("read_analyze", vertical="coding")

        # Query chains
        coding_chains = registry.find_by_vertical("coding")
    """

    _instance: Optional["ChainRegistry"] = None
    _class_lock: threading.Lock = threading.Lock()

    def __init__(self):
        """Initialize the registry."""
        self._chains: Dict[str, Any] = {}
        self._factories: Dict[str, Callable[[], Any]] = {}
        self._metadata: Dict[str, ChainMetadata] = {}
        self._lock = threading.RLock()
        # Batch update support for efficient bulk registration
        self._batch_depth: int = 0
        self._batch_dirty: bool = False
        # Inverted indexes for O(1) discovery
        self._tag_index: Dict[str, Set[str]] = {}  # tag -> {keys}
        self._vertical_index: Dict[str, Set[str]] = {}  # vertical -> {keys}

    @classmethod
    def get_instance(cls) -> "ChainRegistry":
        """Get the singleton instance of ChainRegistry.

        Thread-safe singleton access.

        Returns:
            The global ChainRegistry instance.
        """
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing).

        Creates a fresh registry on next get_instance() call.
        """
        with cls._class_lock:
            cls._instance = None

    @contextmanager
    def batch_update(self) -> Generator[None, None, None]:
        """Batch multiple chain mutations with reduced lock overhead.

        Use when registering many chains at once (e.g., during startup
        or vertical activation) to avoid repeated lock acquisition.

        Example:
            with registry.batch_update():
                registry.register("chain1", chain1, vertical="test")
                registry.register("chain2", chain2, vertical="test")
                registry.register("chain3", chain3, vertical="test")
            # Single lock acquisition for all operations
        """
        with self._lock:
            self._batch_depth += 1
            try:
                yield
            finally:
                self._batch_depth -= 1
                if self._batch_depth == 0:
                    self._batch_dirty = False

    def _update_indexes(self, key: str, metadata: ChainMetadata) -> None:
        """Update inverted indexes for a chain.

        Args:
            key: The chain key in the registry
            metadata: The ChainMetadata to index
        """
        # Index by tags
        for tag in metadata.tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = set()
            self._tag_index[tag].add(key)

        # Index by vertical
        if metadata.vertical:
            if metadata.vertical not in self._vertical_index:
                self._vertical_index[metadata.vertical] = set()
            self._vertical_index[metadata.vertical].add(key)

    def _remove_from_indexes(self, key: str) -> None:
        """Remove a chain key from all inverted indexes."""
        for index in (self._tag_index, self._vertical_index):
            empty_entries = []
            for index_key, keys in index.items():
                keys.discard(key)
                if not keys:
                    empty_entries.append(index_key)
            for index_key in empty_entries:
                del index[index_key]

    def register(
        self,
        name: str,
        chain: Any,
        vertical: Optional[str] = None,
        description: str = "",
        input_type: Optional[str] = None,
        output_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        replace: bool = False,
    ) -> None:
        """Register a chain with optional vertical namespace.

        Args:
            name: Short name for the chain
            chain: The chain object (typically a Runnable)
            vertical: Optional vertical namespace
            description: Human-readable description
            input_type: Optional type hint for chain input
            output_type: Optional type hint for chain output
            tags: Tags for filtering/discovery
            replace: If True, replace existing registration

        Raises:
            ValueError: If name already registered and replace=False
        """
        key = f"{vertical}:{name}" if vertical else name

        # Use conditional lock: only acquire if not in batch mode
        if self._batch_depth > 0:
            self._batch_dirty = True
            if key in self._chains and not replace:
                logger.warning(f"Chain '{key}' already registered, skipping")
                return
            self._remove_from_indexes(key)
            self._chains[key] = chain
            metadata = ChainMetadata(
                name=name,
                vertical=vertical,
                description=description,
                input_type=input_type,
                output_type=output_type,
                tags=tags or [],
                is_factory=False,
            )
            self._metadata[key] = metadata
            self._update_indexes(key, metadata)
        else:
            with self._lock:
                if key in self._chains and not replace:
                    logger.warning(f"Chain '{key}' already registered, skipping")
                    return

                self._remove_from_indexes(key)
                self._chains[key] = chain
                metadata = ChainMetadata(
                    name=name,
                    vertical=vertical,
                    description=description,
                    input_type=input_type,
                    output_type=output_type,
                    tags=tags or [],
                    is_factory=False,
                )
                self._metadata[key] = metadata
                self._update_indexes(key, metadata)
        logger.debug(f"Registered chain: {key}")

    def register_factory(
        self,
        name: str,
        factory: Callable[[], Any],
        vertical: Optional[str] = None,
        description: str = "",
        input_type: Optional[str] = None,
        output_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        version: str = "1.0.0",
        replace: bool = False,
    ) -> None:
        """Register a chain factory for deferred creation.

        Factory functions are called when create() is invoked, allowing
        for lazy initialization of chains that depend on runtime state.

        Args:
            name: Short name for the chain (or "vertical:name" format)
            factory: Callable that returns a chain when invoked
            vertical: Optional vertical namespace
            description: Human-readable description
            input_type: Optional type hint for chain input
            output_type: Optional type hint for chain output
            tags: Tags for filtering/discovery
            version: Version string for the chain
            replace: If True, replace existing registration

        Raises:
            ValueError: If name already registered and replace=False

        Example:
            registry.register_factory(
                "read_analyze",
                lambda: read_tool | analyze_tool,
                vertical="coding",
                description="Read and analyze code",
            )
        """
        # Support "vertical:name" format in name parameter
        if ":" in name and vertical is None:
            vertical, name = name.split(":", 1)

        key = f"{vertical}:{name}" if vertical else name

        with self._lock:
            if (key in self._factories or key in self._chains) and not replace:
                logger.warning(f"Chain/factory '{key}' already registered, skipping")
                return

            self._remove_from_indexes(key)
            self._factories[key] = factory
            metadata = ChainMetadata(
                name=name,
                vertical=vertical,
                description=description,
                input_type=input_type,
                output_type=output_type,
                tags=tags or [],
                is_factory=True,
                version=version,
            )
            self._metadata[key] = metadata
            self._update_indexes(key, metadata)
            logger.debug(f"Registered chain factory: {key}")

    def create(self, name: str, vertical: Optional[str] = None) -> Optional[Any]:
        """Create a chain from a registered factory.

        Invokes the factory function and returns the created chain.
        Does not cache the result - each call creates a fresh chain.

        Args:
            name: Chain name (or "vertical:name" format)
            vertical: Optional vertical namespace

        Returns:
            Created chain object, or None if factory not found

        Raises:
            RuntimeError: If factory execution fails

        Example:
            chain = registry.create("read_analyze", vertical="coding")
        """
        # Support "vertical:name" format
        if ":" in name and vertical is None:
            vertical, name = name.split(":", 1)

        key = f"{vertical}:{name}" if vertical else name

        with self._lock:
            if key in self._factories:
                factory = self._factories[key]
            elif vertical:
                # Try without vertical prefix
                factory = self._factories.get(name)
            else:
                factory = None

        if factory is None:
            logger.warning(f"No factory registered for chain: {key}")
            return None

        try:
            chain = factory()
            logger.debug(f"Created chain from factory: {key}")
            return chain
        except Exception as e:
            logger.error(f"Failed to create chain '{key}': {e}")
            raise RuntimeError(f"Factory execution failed for '{key}': {e}") from e

    def has(self, name: str, vertical: Optional[str] = None) -> bool:
        """Check if a chain or factory is registered.

        Args:
            name: Chain name
            vertical: Optional vertical namespace

        Returns:
            True if registered (either as chain or factory)
        """
        key = f"{vertical}:{name}" if vertical else name
        with self._lock:
            return key in self._chains or key in self._factories

    def unregister(self, name: str, vertical: Optional[str] = None) -> bool:
        """Unregister a chain or factory.

        Args:
            name: Chain name to unregister
            vertical: Optional vertical namespace

        Returns:
            True if unregistered, False if not found
        """
        key = f"{vertical}:{name}" if vertical else name

        with self._lock:
            found = False
            if key in self._chains:
                del self._chains[key]
                found = True
            if key in self._factories:
                del self._factories[key]
                found = True
            if key in self._metadata:
                del self._metadata[key]
            self._remove_from_indexes(key)

            if found:
                logger.debug(f"Unregistered chain: {key}")
            return found

    def get(self, name: str, vertical: Optional[str] = None) -> Optional[Any]:
        """Get a chain by name, optionally scoped to vertical.

        Args:
            name: Chain name to retrieve
            vertical: Optional vertical to scope the search

        Returns:
            Chain object or None if not found
        """
        with self._lock:
            if vertical:
                key = f"{vertical}:{name}"
                if key in self._chains:
                    return self._chains[key]
            return self._chains.get(name)

    def get_metadata(self, name: str, vertical: Optional[str] = None) -> Optional[ChainMetadata]:
        """Get metadata for a chain.

        Args:
            name: Chain name
            vertical: Optional vertical namespace

        Returns:
            ChainMetadata or None if not found
        """
        with self._lock:
            if vertical:
                key = f"{vertical}:{name}"
                if key in self._metadata:
                    return self._metadata[key]
            return self._metadata.get(name)

    def list_chains(self, vertical: Optional[str] = None) -> List[str]:
        """List all registered chain names.

        Args:
            vertical: If provided, only list chains from this vertical

        Returns:
            List of chain names (full keys)
        """
        with self._lock:
            if vertical:
                prefix = f"{vertical}:"
                return [k for k in self._chains.keys() if k.startswith(prefix)]
            return list(self._chains.keys())

    def list_metadata(self, vertical: Optional[str] = None) -> List[ChainMetadata]:
        """List all chain metadata.

        Args:
            vertical: If provided, only list metadata from this vertical

        Returns:
            List of ChainMetadata objects
        """
        with self._lock:
            if vertical:
                keys = self._vertical_index.get(vertical, set())
                return [self._metadata[k] for k in keys if k in self._metadata]
            return list(self._metadata.values())

    def find_by_vertical(self, vertical: str) -> Dict[str, Any]:
        """Get all chains for a specific vertical.

        Args:
            vertical: Vertical name to filter by

        Returns:
            Dict mapping full names to chain objects
        """
        with self._lock:
            keys = self._vertical_index.get(vertical, set())
            return {k: self._chains[k] for k in keys if k in self._chains}

    def find_by_tag(self, tag: str) -> Dict[str, Any]:
        """Find chains with a specific tag.

        Args:
            tag: Tag to filter by

        Returns:
            Dict mapping names to chain objects
        """
        with self._lock:
            keys = self._tag_index.get(tag, set())
            return {k: self._chains[k] for k in keys if k in self._chains}

    def find_by_tags(self, tags: List[str], match_all: bool = False) -> Dict[str, Any]:
        """Find chains matching multiple tags.

        Args:
            tags: List of tags to match
            match_all: If True, match all tags; if False, match any

        Returns:
            Dict mapping names to chain objects
        """
        with self._lock:
            if not tags:
                return self._chains.copy()

            if match_all:
                # Find keys that have all tags (intersection)
                keys = None
                for tag in tags:
                    tag_keys = self._tag_index.get(tag, set())
                    if keys is None:
                        keys = tag_keys.copy()
                    else:
                        keys &= tag_keys
                        if not keys:  # Early exit if no match
                            return {}
                return {k: self._chains[k] for k in (keys or set()) if k in self._chains}
            else:
                # Find keys that have any tag (union)
                keys = set()
                for tag in tags:
                    keys.update(self._tag_index.get(tag, set()))
                return {k: self._chains[k] for k in keys if k in self._chains}

    def clear(self) -> None:
        """Clear all registered chains and factories (for testing)."""
        with self._lock:
            self._chains.clear()
            self._factories.clear()
            self._metadata.clear()
            self._tag_index.clear()
            self._vertical_index.clear()
            logger.debug("Cleared chain registry")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the registry to a dictionary.

        Returns:
            Dict with chain names as keys and metadata dicts as values
        """
        with self._lock:
            return {key: meta.to_dict() for key, meta in self._metadata.items()}

    def list_factories(self, vertical: Optional[str] = None) -> List[str]:
        """List all registered factory names.

        Args:
            vertical: If provided, only list factories from this vertical

        Returns:
            List of factory names (full keys)
        """
        with self._lock:
            if vertical:
                prefix = f"{vertical}:"
                return [k for k in self._factories.keys() if k.startswith(prefix)]
            return list(self._factories.keys())

    def register_from_vertical(
        self,
        vertical_name: str,
        chains: Dict[str, Any],
        replace: bool = True,
    ) -> int:
        """Register multiple chains from a vertical.

        Convenience method for bulk registration with namespace prefixing.

        Args:
            vertical_name: Vertical name for namespace
            chains: Dict mapping chain names to chain objects
            replace: If True, replace existing registrations

        Returns:
            Number of chains registered
        """
        count = 0
        for name, chain in chains.items():
            try:
                self.register(
                    name,
                    chain,
                    vertical=vertical_name,
                    replace=replace,
                )
                count += 1
            except Exception as e:
                logger.warning(f"Failed to register {vertical_name}:{name}: {e}")

        logger.info(f"Registered {count} chains from vertical '{vertical_name}'")
        return count


def get_chain_registry() -> ChainRegistry:
    """Get the global chain registry.

    Thread-safe singleton access.

    Returns:
        Global ChainRegistry instance
    """
    global _registry_instance

    if _registry_instance is None:
        with _registry_lock:
            if _registry_instance is None:
                _registry_instance = ChainRegistry()

    return _registry_instance


def register_chain(
    name: str,
    chain: Any,
    *,
    vertical: Optional[str] = None,
    description: str = "",
    tags: Optional[List[str]] = None,
    replace: bool = False,
) -> None:
    """Register a chain in the global registry.

    Convenience function for quick registration.

    Args:
        name: Chain name
        chain: Chain object
        vertical: Optional vertical namespace
        description: Human-readable description
        tags: Tags for discovery
        replace: Replace existing if present
    """
    get_chain_registry().register(
        name,
        chain,
        vertical=vertical,
        description=description,
        tags=tags,
        replace=replace,
    )


def get_chain(name: str, vertical: Optional[str] = None) -> Optional[Any]:
    """Get a chain from the global registry.

    Args:
        name: Chain name
        vertical: Optional vertical namespace

    Returns:
        Chain object or None
    """
    return get_chain_registry().get(name, vertical=vertical)


def create_chain(name: str, vertical: Optional[str] = None) -> Optional[Any]:
    """Create a chain from a registered factory.

    Convenience function for creating chains from factories.

    Args:
        name: Chain name (or "vertical:name" format)
        vertical: Optional vertical namespace

    Returns:
        Created chain object, or None if factory not found
    """
    return get_chain_registry().create(name, vertical=vertical)


def chain(
    name: str,
    *,
    description: str = "",
    input_type: Optional[str] = None,
    output_type: Optional[str] = None,
    tags: Optional[List[str]] = None,
    version: str = "1.0.0",
    replace: bool = False,
) -> Callable[[F], F]:
    """Decorator for registering a chain factory function.

    The decorated function becomes a factory that is registered in the
    global ChainRegistry. When the chain is requested via create(), the
    factory function is called to create a fresh instance.

    Args:
        name: Chain name (supports "vertical:name" format)
        description: Human-readable description
        input_type: Optional type hint for chain input
        output_type: Optional type hint for chain output
        tags: Tags for filtering/discovery
        version: Version string for the chain
        replace: If True, replace existing registration

    Returns:
        Decorator function

    Example:
        @chain("coding:read_analyze", description="Read and analyze code")
        def read_and_analyze():
            return read_tool | analyze_tool | format_fn

        # Later, create the chain:
        my_chain = create_chain("coding:read_analyze")

        # Or use full API:
        registry = get_chain_registry()
        my_chain = registry.create("read_analyze", vertical="coding")
    """

    def decorator(func: F) -> F:
        # Parse vertical from name if present
        vertical = None
        chain_name = name
        if ":" in name:
            vertical, chain_name = name.split(":", 1)

        # Register the factory
        get_chain_registry().register_factory(
            chain_name,
            func,
            vertical=vertical,
            description=description,
            input_type=input_type,
            output_type=output_type,
            tags=tags,
            version=version,
            replace=replace,
        )

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def reset_chain_registry() -> None:
    """Reset the global chain registry (for testing).

    Creates a fresh registry on next access.
    """
    global _registry_instance
    with _registry_lock:
        _registry_instance = None
    ChainRegistry.reset_instance()


__all__ = [
    "ChainRegistry",
    "ChainMetadata",
    "get_chain_registry",
    "register_chain",
    "get_chain",
    "create_chain",
    "chain",
    "reset_chain_registry",
]
