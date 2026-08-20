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

"""Generic base registry implementations."""

from __future__ import annotations

import logging
import threading
from abc import ABC
from typing import Any, ClassVar, Dict, Generic, Iterator, List, Optional, TypeVar

logger = logging.getLogger(__name__)

K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T", bound="SingletonRegistry")


class BaseRegistry(Generic[K, V]):
    """Generic registry base class for all registry implementations.

    Provides a consistent interface for registering, retrieving, and managing
    items by key. This class can be extended to create specialized registries
    for tools, providers, or any other type of managed object.

    Type Parameters:
        K: The key type (typically str)
        V: The value type (e.g., BaseTool, BaseProvider)

    Example:
        >>> class ToolRegistry(BaseRegistry[str, BaseTool]):
        ...     pass
        >>> registry = ToolRegistry()
        >>> registry.register("my_tool", my_tool_instance)
        >>> registry.get("my_tool")
        <BaseTool instance>

    Attributes:
        _items: Internal dictionary storing registered items
    """

    def __init__(self) -> None:
        """Initialize the registry with an empty items dictionary."""
        self._items: Dict[K, V] = {}

    def register(self, key: K, value: V) -> None:
        """Register an item with the given key.

        Args:
            key: The unique identifier for the item
            value: The item to register

        Note:
            If an item with the same key already exists, it will be overwritten.
        """
        self._items[key] = value

    def get(self, key: K) -> Optional[V]:
        """Get an item by key.

        Args:
            key: The unique identifier for the item

        Returns:
            The registered item, or None if not found
        """
        return self._items.get(key)

    def list_all(self) -> List[K]:
        """List all registered keys.

        Returns:
            A list of all keys in the registry
        """
        return list(self._items.keys())

    def keys(self) -> Iterator[K]:
        """Return an iterator over registry keys.

        Provides dict-like API compatibility for code that expects .keys().

        Returns:
            Iterator over all keys in the registry
        """
        return iter(self._items.keys())

    def unregister(self, key: K) -> bool:
        """Unregister an item by key.

        Args:
            key: The unique identifier for the item to remove

        Returns:
            True if the item was found and removed, False otherwise
        """
        if key in self._items:
            del self._items[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all items from the registry."""
        self._items.clear()

    def __contains__(self, key: K) -> bool:
        """Check if a key exists in the registry.

        Args:
            key: The key to check

        Returns:
            True if the key exists, False otherwise
        """
        return key in self._items

    def __len__(self) -> int:
        """Get the number of items in the registry.

        Returns:
            The count of registered items
        """
        return len(self._items)

    def __iter__(self) -> Iterator[K]:
        """Iterate over registered keys.

        Returns:
            An iterator over the registry keys
        """
        return iter(self._items)

    def values(self) -> List[V]:
        """Get all registered values.

        Returns:
            A list of all registered values
        """
        return list(self._items.values())

    def items(self) -> List[tuple[K, V]]:
        """Get all key-value pairs.

        Returns:
            A list of (key, value) tuples
        """
        return list(self._items.items())


class SingletonRegistry(ABC, Generic[T]):
    """Base class for thread-safe singleton registries.

    This class provides the common singleton infrastructure used by registry
    implementations that need one process-local instance per subclass.
    """

    _instance: ClassVar[Optional["SingletonRegistry"]] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()
    _instantiation_allowed: ClassVar[bool] = True

    def __init__(self) -> None:
        """Initialize the registry.

        Subclasses should call ``super().__init__()`` before initializing their
        own storage.
        """
        cls = type(self)
        if not cls._instantiation_allowed and cls._instance is not None:
            raise RuntimeError(
                f"{cls.__name__} is a singleton. Use {cls.__name__}.get_instance() "
                "to get the singleton instance."
            )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Give every subclass independent singleton state."""
        super().__init_subclass__(**kwargs)
        cls._instance = None
        cls._lock = threading.Lock()
        cls._instantiation_allowed = True

    @classmethod
    def get_instance(cls: type[T]) -> T:
        """Get the singleton instance for this registry subclass."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    was_allowed = cls._instantiation_allowed
                    cls._instantiation_allowed = True
                    try:
                        cls._instance = cls()  # type: ignore[assignment]
                        logger.debug("%s singleton instance created", cls.__name__)
                    finally:
                        cls._instantiation_allowed = was_allowed
        return cls._instance  # type: ignore[return-value]

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance, primarily for test isolation."""
        with cls._lock:
            cls._instance = None
            logger.debug("%s singleton instance reset", cls.__name__)

    @classmethod
    def is_initialized(cls) -> bool:
        """Return whether the singleton instance has been created."""
        return cls._instance is not None


class ItemRegistry(SingletonRegistry[T], Generic[T]):
    """Base class for singleton registries that store items by name."""

    def __init__(self) -> None:
        """Initialize the item registry with empty storage."""
        super().__init__()
        self._items: Dict[str, Any] = {}
        self._items_lock: threading.RLock = threading.RLock()

    def register(self, name: str, item: Any) -> None:
        """Register an item by name."""
        with self._items_lock:
            self._items[name] = item
            logger.debug("%s: Registered '%s'", type(self).__name__, name)

    def unregister(self, name: str) -> bool:
        """Unregister an item by name."""
        with self._items_lock:
            if name in self._items:
                del self._items[name]
                logger.debug("%s: Unregistered '%s'", type(self).__name__, name)
                return True
            return False

    def get(self, name: str) -> Optional[Any]:
        """Get an item by name."""
        with self._items_lock:
            return self._items.get(name)

    def contains(self, name: str) -> bool:
        """Check if an item is registered."""
        with self._items_lock:
            return name in self._items

    def list_names(self) -> List[str]:
        """Get all registered item names."""
        with self._items_lock:
            return list(self._items.keys())

    def list_items(self) -> List[Any]:
        """Get all registered items."""
        with self._items_lock:
            return list(self._items.values())

    def count(self) -> int:
        """Get count of registered items."""
        with self._items_lock:
            return len(self._items)

    def clear(self) -> int:
        """Clear all registered items and return the number cleared."""
        with self._items_lock:
            count = len(self._items)
            self._items.clear()
            logger.debug("%s: Cleared %s items", type(self).__name__, count)
            return count
