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

"""Shared base for flat, vertical-namespaced registries (internal).

Not public framework API: only ``ChainRegistry`` and ``PersonaRegistry``
subclass it, and their public surfaces are unchanged.

``ChainRegistry`` and ``PersonaRegistry`` are both flat registries keyed by
``"{vertical}:{name}"`` with a factory table, inverted indexes and a
per-class singleton. This module owns that shared machinery so each concrete
registry only declares its value type, its index terms and its
domain-specific finders.

Design:
- One singleton per concrete subclass (``__init_subclass__`` gives each
  subclass its own ``_instance`` slot and class lock).
- One re-entrant lock guards every mutation; ``batch_update()`` simply holds
  it across several calls, so other threads block for the whole batch.
- Inverted indexes are driven by the ``_index_terms`` hook, which maps an
  index name to the terms an entry contributes to it.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generator,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
)

T = TypeVar("T")
RegistryT = TypeVar("RegistryT", bound="NamespacedRegistry[Any]")

_module_logger = logging.getLogger(__name__)


def make_key(name: str, vertical: Optional[str] = None) -> str:
    """Build a registry key from a short name and optional vertical.

    Args:
        name: Short entry name
        vertical: Optional vertical namespace

    Returns:
        ``"vertical:name"`` when a vertical is given, otherwise ``name``
    """
    return f"{vertical}:{name}" if vertical else name


def split_qualified_name(name: str, vertical: Optional[str] = None) -> Tuple[Optional[str], str]:
    """Split a ``"vertical:name"`` reference when no vertical is given.

    Args:
        name: Short name or ``"vertical:name"`` reference
        vertical: Explicit vertical; when set, ``name`` is used verbatim

    Returns:
        Tuple of (vertical, name)
    """
    if ":" in name and vertical is None:
        vertical, name = name.split(":", 1)
    return vertical, name


class NamespacedRegistry(Generic[T]):
    """Generic thread-safe registry of entries namespaced by vertical.

    Subclasses set ``_KIND``/``_KIND_PLURAL`` (used in log messages),
    ``_INDEX_NAMES`` (the inverted indexes to maintain) and ``_logger``
    (so log records keep the concrete module's logger name), and implement
    ``_index_terms``.

    Attributes:
        _items: Direct entries keyed by full key
        _factories: Deferred-creation factories keyed by full key
        _indexes: index name -> term -> set of full keys
    """

    _instance: Optional["NamespacedRegistry[Any]"] = None
    _class_lock: threading.Lock = threading.Lock()

    _KIND: ClassVar[str] = "Entry"
    _KIND_PLURAL: ClassVar[str] = "entries"
    _INDEX_NAMES: ClassVar[Tuple[str, ...]] = ()
    _logger: ClassVar[logging.Logger] = _module_logger

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Give every subclass its own singleton slot and class lock."""
        super().__init_subclass__(**kwargs)
        cls._instance = None
        cls._class_lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize empty storage, indexes and the re-entrant lock."""
        self._items: Dict[str, T] = {}
        self._factories: Dict[str, Callable[[], Any]] = {}
        self._lock = threading.RLock()
        self._indexes: Dict[str, Dict[str, Set[str]]] = {name: {} for name in self._INDEX_NAMES}

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls: Type[RegistryT]) -> RegistryT:
        """Get the singleton instance of this registry class.

        Thread-safe singleton access; each subclass has its own instance.

        Returns:
            The global instance for ``cls``.
        """
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance  # type: ignore[return-value]

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing).

        Creates a fresh registry on next get_instance() call.
        """
        with cls._class_lock:
            cls._instance = None

    # ------------------------------------------------------------------
    # Locking
    # ------------------------------------------------------------------

    @contextmanager
    def batch_update(self) -> Generator[None, None, None]:
        """Hold the registry lock across several mutations.

        Use when registering many entries at once (e.g., during startup or
        vertical activation). The lock is re-entrant, so calls made inside
        the block by the same thread proceed; other threads block until the
        outermost batch exits. Do not wait inside a batch on another thread
        that registers into the same registry: it would deadlock.

        Example:
            with registry.batch_update():
                registry.register("a", a, vertical="test")
                registry.register("b", b, vertical="test")
        """
        with self._lock:
            yield

    # ------------------------------------------------------------------
    # Indexes
    # ------------------------------------------------------------------

    def _index_terms(self, source: Any) -> Mapping[str, Iterable[str]]:
        """Map each index name to the terms ``source`` contributes.

        Args:
            source: The object describing an entry (value or metadata)

        Returns:
            Mapping of index name (from ``_INDEX_NAMES``) to terms
        """
        return {}

    def _add_to_indexes(self, key: str, source: Any) -> None:
        """Add ``key`` to every index term ``source`` contributes."""
        for index_name, terms in self._index_terms(source).items():
            index = self._indexes[index_name]
            for term in terms:
                index.setdefault(term, set()).add(key)

    def _remove_from_indexes(self, key: str) -> None:
        """Remove ``key`` from all indexes, dropping emptied terms."""
        for index in self._indexes.values():
            empty_terms = []
            for term, keys in index.items():
                keys.discard(key)
                if not keys:
                    empty_terms.append(term)
            for term in empty_terms:
                del index[term]

    def _index_keys(self, index_name: str, term: str) -> Set[str]:
        """Return the keys indexed under ``term`` (caller holds the lock)."""
        return self._indexes[index_name].get(term, set())

    def _match_index(self, index_name: str, terms: List[str], match_all: bool) -> Set[str]:
        """Keys matching any (or all) of ``terms`` in one index.

        Caller holds the lock and handles the empty-``terms`` case.

        Args:
            index_name: Index to search
            terms: Terms to match
            match_all: Intersect when True, union otherwise

        Returns:
            Matching keys
        """
        if match_all:
            keys: Optional[Set[str]] = None
            for term in terms:
                term_keys = self._index_keys(index_name, term)
                if keys is None:
                    keys = set(term_keys)
                else:
                    keys &= term_keys
                if not keys:
                    return set()
            return keys or set()
        matched: Set[str] = set()
        for term in terms:
            matched.update(self._index_keys(index_name, term))
        return matched

    def _items_dict(self, keys: Iterable[str]) -> Dict[str, T]:
        """Project keys to ``{key: item}`` for direct entries."""
        return {k: self._items[k] for k in keys if k in self._items}

    def _items_list(self, keys: Iterable[str]) -> List[T]:
        """Project keys to a list of direct entries."""
        return [self._items[k] for k in keys if k in self._items]

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def _put(self, key: str, item: T, *, replace: bool, index_source: Any) -> bool:
        """Store a direct entry and re-index it under the registry lock.

        Args:
            key: Full registry key
            item: Entry to store
            replace: Overwrite an existing direct entry when True
            index_source: Object passed to ``_index_terms``

        Returns:
            True if stored, False if skipped as a duplicate
        """
        with self._lock:
            if key in self._items and not replace:
                self._logger.warning(f"{self._KIND} '{key}' already registered, skipping")
                return False
            self._remove_from_indexes(key)
            self._items[key] = item
            self._add_to_indexes(key, index_source)
            return True

    def _put_factory(
        self,
        key: str,
        factory: Callable[[], Any],
        *,
        replace: bool,
        duplicate_level: int = logging.WARNING,
    ) -> bool:
        """Store a factory under the registry lock.

        Args:
            key: Full registry key
            factory: Zero-argument callable creating the entry
            replace: Overwrite an existing entry or factory when True
            duplicate_level: Log level for a skipped duplicate

        Returns:
            True if stored, False if skipped as a duplicate
        """
        with self._lock:
            if (key in self._factories or key in self._items) and not replace:
                self._logger.log(
                    duplicate_level,
                    f"{self._KIND}/factory '{key}' already registered, skipping",
                )
                return False
            self._factories[key] = factory
            return True

    def _forget(self, key: str) -> None:
        """Hook to drop subclass-owned per-key state on unregister."""

    def unregister(self, name: str, vertical: Optional[str] = None) -> bool:
        """Unregister an entry and/or factory.

        Args:
            name: Entry name to unregister
            vertical: Optional vertical namespace

        Returns:
            True if unregistered, False if not found
        """
        key = make_key(name, vertical)
        with self._lock:
            found = key in self._items
            if found:
                del self._items[key]
            if key in self._factories:
                del self._factories[key]
                found = True
            self._forget(key)
            self._remove_from_indexes(key)
            if found:
                self._logger.debug(f"Unregistered {self._KIND.lower()}: {key}")
            return found

    def clear(self) -> None:
        """Clear all entries, factories and indexes (for testing)."""
        with self._lock:
            self._items.clear()
            self._factories.clear()
            for index in self._indexes.values():
                index.clear()
            self._clear_extra()
            self._logger.debug(f"Cleared {self._KIND.lower()} registry")

    def _clear_extra(self) -> None:
        """Hook to clear subclass-owned state in ``clear()``."""

    def _register_from_vertical(
        self,
        vertical_name: str,
        items: Mapping[str, Any],
        register: Callable[[str, Any], None],
    ) -> int:
        """Register many entries for a vertical via ``register``.

        Args:
            vertical_name: Vertical name for the namespace
            items: Mapping of short name to entry
            register: Callable registering one ``(name, entry)`` pair

        Returns:
            Number of entries for which ``register`` did not raise (an entry
            skipped as a duplicate with ``replace=False`` still counts)
        """
        count = 0
        for name, item in items.items():
            try:
                register(name, item)
                count += 1
            except Exception as e:
                self._logger.warning(f"Failed to register {vertical_name}:{name}: {e}")

        self._logger.info(f"Registered {count} {self._KIND_PLURAL} from vertical '{vertical_name}'")
        return count

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def _resolve_factory(
        self, name: str, vertical: Optional[str]
    ) -> Tuple[str, Optional[str], Optional[Callable[[], Any]]]:
        """Find a factory, falling back from ``vertical:name`` to ``name``.

        Args:
            name: Entry name (or ``"vertical:name"``)
            vertical: Optional vertical namespace

        Returns:
            Tuple of (full key, vertical, factory or None)
        """
        vertical, name = split_qualified_name(name, vertical)
        key = make_key(name, vertical)
        with self._lock:
            if key in self._factories:
                factory: Optional[Callable[[], Any]] = self._factories[key]
            elif vertical:
                factory = self._factories.get(name)
            else:
                factory = None
        return key, vertical, factory

    def has_factory(self, name: str, vertical: Optional[str] = None) -> bool:
        """Check if a factory is registered under exactly ``vertical:name``.

        Args:
            name: Entry name
            vertical: Optional vertical namespace

        Returns:
            True if a factory exists under the exact key (no bare-name fallback)
        """
        with self._lock:
            return make_key(name, vertical) in self._factories

    def has(self, name: str, vertical: Optional[str] = None) -> bool:
        """Check if an entry or factory is registered under the exact key.

        Args:
            name: Entry name
            vertical: Optional vertical namespace

        Returns:
            True if registered (either as entry or factory)
        """
        key = make_key(name, vertical)
        with self._lock:
            return key in self._items or key in self._factories

    def get(self, name: str, vertical: Optional[str] = None) -> Optional[T]:
        """Get an entry, trying ``vertical:name`` then the bare ``name``.

        Args:
            name: Entry name to retrieve
            vertical: Optional vertical to scope the search

        Returns:
            Entry or None if not found
        """
        with self._lock:
            if vertical:
                key = make_key(name, vertical)
                if key in self._items:
                    return self._items[key]
            return self._items.get(name)

    def _list_keys(self, table: Mapping[str, Any], vertical: Optional[str]) -> List[str]:
        """List keys of ``table``, optionally filtered to one vertical."""
        with self._lock:
            if vertical:
                prefix = f"{vertical}:"
                return [k for k in table.keys() if k.startswith(prefix)]
            return list(table.keys())

    def list_factories(self, vertical: Optional[str] = None) -> List[str]:
        """List all registered factory names.

        Args:
            vertical: If provided, only list factories from this vertical

        Returns:
            List of factory names (full keys)
        """
        return self._list_keys(self._factories, vertical)


__all__ = ["NamespacedRegistry", "make_key", "split_qualified_name"]
