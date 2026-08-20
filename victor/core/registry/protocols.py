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

"""Protocol definitions for registry implementations."""

from typing import List, Optional, Protocol, TypeVar, runtime_checkable

K = TypeVar("K", contravariant=True)
V = TypeVar("V", covariant=True)


@runtime_checkable
class IRegistry(Protocol[K, V]):
    """Protocol for registry implementations.

    A registry provides a simple key-value store interface for registering,
    retrieving, and managing items. This protocol enables consistent behavior
    across different registry implementations (tools, providers, etc.).

    Type Parameters:
        K: The key type (contravariant - accepts subtypes of K)
        V: The value type (covariant - returns subtypes of V)

    Example:
        >>> class MyRegistry(IRegistry[str, MyItem]):
        ...     def register(self, key: str, value: MyItem) -> None: ...
        ...     def get(self, key: str) -> Optional[MyItem]: ...
        ...     def list_all(self) -> List[str]: ...
        ...     def unregister(self, key: str) -> bool: ...
        ...     def clear(self) -> None: ...
    """

    def register(self, key: K, value: V) -> None:
        """Register an item with the given key.

        Args:
            key: The unique identifier for the item
            value: The item to register

        Note:
            If an item with the same key already exists, it will be overwritten.
        """
        ...

    def get(self, key: K) -> Optional[V]:
        """Get an item by key.

        Args:
            key: The unique identifier for the item

        Returns:
            The registered item, or None if not found
        """
        ...

    def list_all(self) -> List[K]:
        """List all registered keys.

        Returns:
            A list of all keys in the registry
        """
        ...

    def unregister(self, key: K) -> bool:
        """Unregister an item by key.

        Args:
            key: The unique identifier for the item to remove

        Returns:
            True if the item was found and removed, False otherwise
        """
        ...

    def clear(self) -> None:
        """Clear all items from the registry."""
        ...
