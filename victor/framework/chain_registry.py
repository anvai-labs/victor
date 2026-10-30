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
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Set, TypeVar

from victor.framework.namespaced_registry import (
    NamespacedRegistry,
    make_key,
    split_qualified_name,
)

F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


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


class ChainRegistry(NamespacedRegistry[Any]):
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

    _KIND = "Chain"
    _KIND_PLURAL = "chains"
    _INDEX_NAMES = ("tag", "vertical")
    _logger = logger

    def __init__(self) -> None:
        """Initialize the registry."""
        super().__init__()
        self._chains: Dict[str, Any] = self._items
        self._metadata: Dict[str, ChainMetadata] = {}
        # Inverted indexes for O(1) discovery (views onto the base indexes)
        self._tag_index: Dict[str, Set[str]] = self._indexes["tag"]
        self._vertical_index: Dict[str, Set[str]] = self._indexes["vertical"]

    def _index_terms(self, source: Any) -> Mapping[str, Iterable[str]]:
        """Index chain metadata by tags and vertical."""
        metadata: ChainMetadata = source
        return {
            "tag": metadata.tags,
            "vertical": [metadata.vertical] if metadata.vertical else [],
        }

    def _forget(self, key: str) -> None:
        """Drop chain metadata on unregister."""
        self._metadata.pop(key, None)

    def _clear_extra(self) -> None:
        """Clear chain metadata on clear()."""
        self._metadata.clear()

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
        """
        key = make_key(name, vertical)
        metadata = ChainMetadata(
            name=name,
            vertical=vertical,
            description=description,
            input_type=input_type,
            output_type=output_type,
            tags=tags or [],
            is_factory=False,
        )
        with self._lock:
            if not self._put(key, chain, replace=replace, index_source=metadata):
                return
            self._metadata[key] = metadata
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

        Example:
            registry.register_factory(
                "read_analyze",
                lambda: read_tool | analyze_tool,
                vertical="coding",
                description="Read and analyze code",
            )
        """
        vertical, name = split_qualified_name(name, vertical)
        key = make_key(name, vertical)
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
        with self._lock:
            if not self._put_factory(key, factory, replace=replace):
                return
            self._remove_from_indexes(key)
            self._metadata[key] = metadata
            self._add_to_indexes(key, metadata)
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
        key, _vertical, factory = self._resolve_factory(name, vertical)

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
                key = make_key(name, vertical)
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
        return self._list_keys(self._chains, vertical)

    def list_metadata(self, vertical: Optional[str] = None) -> List[ChainMetadata]:
        """List all chain metadata.

        Args:
            vertical: If provided, only list metadata from this vertical

        Returns:
            List of ChainMetadata objects
        """
        with self._lock:
            if vertical:
                keys = self._index_keys("vertical", vertical)
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
            return self._items_dict(self._index_keys("vertical", vertical))

    def find_by_tag(self, tag: str) -> Dict[str, Any]:
        """Find chains with a specific tag.

        Args:
            tag: Tag to filter by

        Returns:
            Dict mapping names to chain objects
        """
        with self._lock:
            return self._items_dict(self._index_keys("tag", tag))

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
            return self._items_dict(self._match_index("tag", tags, match_all))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the registry to a dictionary.

        Returns:
            Dict with chain names as keys and metadata dicts as values
        """
        with self._lock:
            return {key: meta.to_dict() for key, meta in self._metadata.items()}

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
        return self._register_from_vertical(
            vertical_name,
            chains,
            lambda name, chain: self.register(name, chain, vertical=vertical_name, replace=replace),
        )


def get_chain_registry() -> ChainRegistry:
    """Get the global chain registry.

    Thread-safe singleton access; the same object as
    ``ChainRegistry.get_instance()``.

    Returns:
        Global ChainRegistry instance
    """
    return ChainRegistry.get_instance()


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
