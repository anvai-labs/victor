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

"""Repository pattern for data access abstraction.

This module provides a comprehensive repository implementation that:
- Abstracts data access from business logic
- Provides collection-like interface for domain objects
- Supports multiple storage backends
- Enables unit testing through in-memory implementations

Design Patterns:
- Repository: Collection-like interface for aggregate persistence
- Unit of Work: Transaction management (see unit_of_work.py)
- Specification: Query building (see specification.py)
- Identity Map: Prevent duplicate object loading

Example:
    from victor.core.repository import (
        Repository,
        InMemoryRepository,
        Entity,
    )

    # Define entity
    @dataclass
    class User(Entity):
        name: str
        email: str

    # Use repository
    repo = InMemoryRepository[User]()
    user = User(id="user-1", name="Alice", email="alice@example.com")
    await repo.add(user)

    # Query
    found = await repo.get("user-1")
    all_users = await repo.list()
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)

from victor.core.event_sourcing import ConcurrencyError

logger = logging.getLogger(__name__)


# =============================================================================
# Entity Base
# =============================================================================


@dataclass
class Entity:
    """Base class for all domain entities.

    Entities have identity and can be persisted in repositories.
    Subclasses should define domain-specific attributes.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = 1

    def __eq__(self, other: object) -> bool:
        """Entities are equal if they have the same id."""
        if not isinstance(other, Entity):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        """Hash by id for use in sets/dicts."""
        return hash(self.id)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize entity to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        """Deserialize entity from dictionary."""
        # Convert datetime strings back to datetime objects
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data and isinstance(data["updated_at"], str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)

    def touch(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now(timezone.utc)

    def increment_version(self) -> None:
        """Increment the version number."""
        self.version += 1


T = TypeVar("T", bound=Entity)


# =============================================================================
# Specification Pattern
# =============================================================================


@runtime_checkable
class Specification(Protocol[T]):
    """Protocol for query specifications.

    Specifications encapsulate query criteria and can be combined
    using boolean logic (and, or, not).
    """

    def is_satisfied_by(self, entity: T) -> bool:
        """Check if entity satisfies this specification."""
        ...

    def to_query(self) -> Dict[str, Any]:
        """Convert specification to query dict for storage backends."""
        ...


@dataclass
class BaseSpecification(Generic[T]):
    """Base class for specifications with combinable logic."""

    def is_satisfied_by(self, entity: T) -> bool:
        """Check if entity satisfies this specification."""
        raise NotImplementedError

    def to_query(self) -> Dict[str, Any]:
        """Convert to query dictionary."""
        return {}

    def __and__(self, other: "BaseSpecification[T]") -> "AndSpecification[T]":
        """Combine with AND."""
        return AndSpecification(self, other)

    def __or__(self, other: "BaseSpecification[T]") -> "OrSpecification[T]":
        """Combine with OR."""
        return OrSpecification(self, other)

    def __invert__(self) -> "NotSpecification[T]":
        """Negate specification."""
        return NotSpecification(self)


@dataclass
class AndSpecification(BaseSpecification[T]):
    """Combines two specifications with AND logic."""

    left: BaseSpecification[T]
    right: BaseSpecification[T]

    def is_satisfied_by(self, entity: T) -> bool:
        return self.left.is_satisfied_by(entity) and self.right.is_satisfied_by(entity)

    def to_query(self) -> Dict[str, Any]:
        return {"$and": [self.left.to_query(), self.right.to_query()]}


@dataclass
class OrSpecification(BaseSpecification[T]):
    """Combines two specifications with OR logic."""

    left: BaseSpecification[T]
    right: BaseSpecification[T]

    def is_satisfied_by(self, entity: T) -> bool:
        return self.left.is_satisfied_by(entity) or self.right.is_satisfied_by(entity)

    def to_query(self) -> Dict[str, Any]:
        return {"$or": [self.left.to_query(), self.right.to_query()]}


@dataclass
class NotSpecification(BaseSpecification[T]):
    """Negates a specification."""

    spec: BaseSpecification[T]

    def is_satisfied_by(self, entity: T) -> bool:
        return not self.spec.is_satisfied_by(entity)

    def to_query(self) -> Dict[str, Any]:
        return {"$not": self.spec.to_query()}


@dataclass
class AttributeSpecification(BaseSpecification[T]):
    """Specification that checks an attribute value."""

    attribute: str
    value: Any
    operator: str = "eq"  # eq, ne, gt, lt, gte, lte, in, contains

    def is_satisfied_by(self, entity: T) -> bool:
        actual = getattr(entity, self.attribute, None)

        if self.operator == "eq":
            return actual == self.value
        elif self.operator == "ne":
            return actual != self.value
        elif self.operator == "gt":
            return actual > self.value
        elif self.operator == "lt":
            return actual < self.value
        elif self.operator == "gte":
            return actual >= self.value
        elif self.operator == "lte":
            return actual <= self.value
        elif self.operator == "in":
            return actual in self.value
        elif self.operator == "contains":
            return self.value in actual if actual else False
        return False

    def to_query(self) -> Dict[str, Any]:
        op_map = {
            "eq": "$eq",
            "ne": "$ne",
            "gt": "$gt",
            "lt": "$lt",
            "gte": "$gte",
            "lte": "$lte",
            "in": "$in",
            "contains": "$contains",
        }
        return {self.attribute: {op_map.get(self.operator, "$eq"): self.value}}


# =============================================================================
# Repository Protocol
# =============================================================================


class Repository(ABC, Generic[T]):
    """Abstract base class for repositories.

    Provides a collection-like interface for persisting and retrieving
    domain entities. Implementations should handle specific storage backends.
    """

    @abstractmethod
    async def add(self, entity: T) -> None:
        """Add a new entity to the repository.

        Args:
            entity: Entity to add

        Raises:
            EntityExistsError: If entity with same id already exists
        """
        ...

    @abstractmethod
    async def get(self, id: str) -> Optional[T]:
        """Get an entity by its id.

        Args:
            id: Entity identifier

        Returns:
            Entity if found, None otherwise
        """
        ...

    @abstractmethod
    async def update(self, entity: T) -> None:
        """Update an existing entity.

        Args:
            entity: Entity to update

        Raises:
            EntityNotFoundError: If entity doesn't exist
            ConcurrencyError: If version conflict detected
        """
        ...

    @abstractmethod
    async def delete(self, id: str) -> None:
        """Delete an entity by its id.

        Args:
            id: Entity identifier

        Raises:
            EntityNotFoundError: If entity doesn't exist
        """
        ...

    @abstractmethod
    async def list(
        self,
        specification: Optional[BaseSpecification[T]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[T]:
        """List entities matching specification.

        Args:
            specification: Optional filter criteria
            limit: Maximum number of entities to return
            offset: Number of entities to skip

        Returns:
            List of matching entities
        """
        ...

    @abstractmethod
    async def count(
        self,
        specification: Optional[BaseSpecification[T]] = None,
    ) -> int:
        """Count entities matching specification.

        Args:
            specification: Optional filter criteria

        Returns:
            Number of matching entities
        """
        ...

    @abstractmethod
    async def exists(self, id: str) -> bool:
        """Check if entity exists.

        Args:
            id: Entity identifier

        Returns:
            True if entity exists
        """
        ...

    async def get_or_raise(self, id: str) -> T:
        """Get entity or raise exception.

        Args:
            id: Entity identifier

        Returns:
            Entity

        Raises:
            EntityNotFoundError: If entity doesn't exist
        """
        entity = await self.get(id)
        if entity is None:
            raise EntityNotFoundError(id)
        return entity

    async def add_many(self, entities: List[T]) -> None:
        """Add multiple entities.

        Args:
            entities: Entities to add
        """
        for entity in entities:
            await self.add(entity)

    async def delete_many(self, ids: List[str]) -> None:
        """Delete multiple entities.

        Args:
            ids: Entity identifiers
        """
        for id in ids:
            await self.delete(id)


# =============================================================================
# Exceptions
# =============================================================================


class RepositoryError(Exception):
    """Base exception for repository errors."""

    pass


class EntityNotFoundError(RepositoryError):
    """Raised when entity is not found."""

    def __init__(self, entity_id: str):
        self.entity_id = entity_id
        super().__init__(f"Entity not found: {entity_id}")


class EntityExistsError(RepositoryError):
    """Raised when entity already exists."""

    def __init__(self, entity_id: str):
        self.entity_id = entity_id
        super().__init__(f"Entity already exists: {entity_id}")


# =============================================================================
# In-Memory Implementation
# =============================================================================


class InMemoryRepository(Repository[T], Generic[T]):
    """In-memory repository for testing and development.

    Uses an identity map to prevent duplicate object loading.
    Supports optimistic concurrency control.
    """

    def __init__(self, entity_class: Optional[Type[T]] = None):
        """Initialize in-memory repository.

        Args:
            entity_class: Optional entity class for type hints
        """
        self._store: Dict[str, T] = {}
        self._entity_class = entity_class
        self._lock = asyncio.Lock()

    async def add(self, entity: T) -> None:
        """Add entity to repository."""
        async with self._lock:
            if entity.id in self._store:
                raise EntityExistsError(entity.id)
            self._store[entity.id] = entity
            logger.debug(f"Added entity {entity.id}")

    async def get(self, id: str) -> Optional[T]:
        """Get entity by id."""
        return self._store.get(id)

    async def update(self, entity: T) -> None:
        """Update entity with optimistic locking."""
        async with self._lock:
            existing = self._store.get(entity.id)
            if existing is None:
                raise EntityNotFoundError(entity.id)

            # Check version for optimistic concurrency
            if existing.version != entity.version:
                raise ConcurrencyError(entity.id, entity.version, existing.version)

            entity.touch()
            entity.increment_version()
            self._store[entity.id] = entity
            logger.debug(f"Updated entity {entity.id}")

    async def delete(self, id: str) -> None:
        """Delete entity by id."""
        async with self._lock:
            if id not in self._store:
                raise EntityNotFoundError(id)
            del self._store[id]
            logger.debug(f"Deleted entity {id}")

    async def list(
        self,
        specification: Optional[BaseSpecification[T]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[T]:
        """List entities matching specification."""
        entities = list(self._store.values())

        if specification:
            entities = [e for e in entities if specification.is_satisfied_by(e)]

        return entities[offset : offset + limit]

    async def count(
        self,
        specification: Optional[BaseSpecification[T]] = None,
    ) -> int:
        """Count entities matching specification."""
        if specification:
            return sum(1 for e in self._store.values() if specification.is_satisfied_by(e))
        return len(self._store)

    async def exists(self, id: str) -> bool:
        """Check if entity exists."""
        return id in self._store

    def clear(self) -> None:
        """Clear all entities (for testing)."""
        self._store.clear()


# =============================================================================
# SQLite Implementation
# =============================================================================


class SQLiteRepository(Repository[T], Generic[T]):
    """SQLite-backed repository for persistent storage.

    Provides durable storage with transaction support and
    optimistic concurrency control.
    """

    def __init__(
        self,
        db_path: Union[str, Path],
        table_name: str,
        entity_class: Type[T],
        create_table: bool = True,
    ):
        """Initialize SQLite repository.

        Args:
            db_path: Path to SQLite database
            table_name: Name of the table to use
            entity_class: Entity class for deserialization
            create_table: Whether to create table if not exists
        """
        self._db_path = str(db_path)
        self._table_name = table_name
        self._entity_class = entity_class
        self._lock = asyncio.Lock()

        if create_table:
            self._create_table()

    def _create_table(self) -> None:
        """Create the repository table if it doesn't exist."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._table_name} (
                    id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """)
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self._table_name}_updated_at
                ON {self._table_name}(updated_at)
                """)

    async def add(self, entity: T) -> None:
        """Add entity to repository."""
        async with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                # Check if exists
                cursor = conn.execute(
                    f"SELECT 1 FROM {self._table_name} WHERE id = ?",
                    (entity.id,),
                )
                if cursor.fetchone():
                    raise EntityExistsError(entity.id)

                data = json.dumps(entity.to_dict())
                conn.execute(
                    f"""
                    INSERT INTO {self._table_name}
                    (id, data, version, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        entity.id,
                        data,
                        entity.version,
                        entity.created_at.isoformat(),
                        entity.updated_at.isoformat(),
                    ),
                )
                logger.debug(f"Added entity {entity.id}")

    async def get(self, id: str) -> Optional[T]:
        """Get entity by id."""
        async with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.execute(
                    f"SELECT data FROM {self._table_name} WHERE id = ?",
                    (id,),
                )
                row = cursor.fetchone()
                if row:
                    data = json.loads(row[0])
                    return self._entity_class.from_dict(data)
                return None

    async def update(self, entity: T) -> None:
        """Update entity with optimistic locking."""
        async with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                # Get current version
                cursor = conn.execute(
                    f"SELECT version FROM {self._table_name} WHERE id = ?",
                    (entity.id,),
                )
                row = cursor.fetchone()
                if row is None:
                    raise EntityNotFoundError(entity.id)

                current_version = row[0]
                if current_version != entity.version:
                    raise ConcurrencyError(entity.id, entity.version, current_version)

                entity.touch()
                entity.increment_version()
                data = json.dumps(entity.to_dict())

                conn.execute(
                    f"""
                    UPDATE {self._table_name}
                    SET data = ?, version = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (data, entity.version, entity.updated_at.isoformat(), entity.id),
                )
                logger.debug(f"Updated entity {entity.id}")

    async def delete(self, id: str) -> None:
        """Delete entity by id."""
        async with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.execute(
                    f"SELECT 1 FROM {self._table_name} WHERE id = ?",
                    (id,),
                )
                if not cursor.fetchone():
                    raise EntityNotFoundError(id)

                conn.execute(
                    f"DELETE FROM {self._table_name} WHERE id = ?",
                    (id,),
                )
                logger.debug(f"Deleted entity {id}")

    async def list(
        self,
        specification: Optional[BaseSpecification[T]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[T]:
        """List entities matching specification."""
        async with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.execute(
                    f"SELECT data FROM {self._table_name} ORDER BY updated_at DESC"
                )

                entities = []
                for row in cursor:
                    data = json.loads(row[0])
                    entity = self._entity_class.from_dict(data)

                    if specification is None or specification.is_satisfied_by(entity):
                        entities.append(entity)

                return entities[offset : offset + limit]

    async def count(
        self,
        specification: Optional[BaseSpecification[T]] = None,
    ) -> int:
        """Count entities matching specification."""
        if specification is None:
            async with self._lock:
                with sqlite3.connect(self._db_path) as conn:
                    cursor = conn.execute(f"SELECT COUNT(*) FROM {self._table_name}")
                    return cursor.fetchone()[0]

        # For specifications, we need to filter in Python
        entities = await self.list(specification, limit=10000)
        return len(entities)

    async def exists(self, id: str) -> bool:
        """Check if entity exists."""
        async with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.execute(
                    f"SELECT 1 FROM {self._table_name} WHERE id = ?",
                    (id,),
                )
                return cursor.fetchone() is not None


# =============================================================================
# Cached Repository
# =============================================================================


class CachedRepository(Repository[T], Generic[T]):
    """Repository decorator that adds caching.

    Uses a two-level cache (identity map + TTL cache) to reduce
    database lookups.
    """

    def __init__(
        self,
        repository: Repository[T],
        cache_ttl: int = 300,  # 5 minutes
        max_cache_size: int = 1000,
    ):
        """Initialize cached repository.

        Args:
            repository: Underlying repository to cache
            cache_ttl: Cache time-to-live in seconds
            max_cache_size: Maximum entities to cache
        """
        self._repository = repository
        self._cache: Dict[str, tuple[T, float]] = {}
        self._cache_ttl = cache_ttl
        self._max_cache_size = max_cache_size
        self._lock = asyncio.Lock()

    def _is_cached(self, id: str) -> bool:
        """Check if entity is in cache and not expired."""
        if id not in self._cache:
            return False

        _, cached_at = self._cache[id]
        import time

        if time.time() - cached_at > self._cache_ttl:
            del self._cache[id]
            return False

        return True

    def _put_in_cache(self, entity: T) -> None:
        """Add entity to cache."""
        import time

        # Evict if cache is full
        if len(self._cache) >= self._max_cache_size:
            # Remove oldest entries
            sorted_entries = sorted(self._cache.items(), key=lambda x: x[1][1])
            for id, _ in sorted_entries[: len(sorted_entries) // 2]:
                del self._cache[id]

        self._cache[entity.id] = (entity, time.time())

    async def add(self, entity: T) -> None:
        """Add entity and cache it."""
        await self._repository.add(entity)
        self._put_in_cache(entity)

    async def get(self, id: str) -> Optional[T]:
        """Get entity from cache or repository."""
        if self._is_cached(id):
            return self._cache[id][0]

        entity = await self._repository.get(id)
        if entity:
            self._put_in_cache(entity)
        return entity

    async def update(self, entity: T) -> None:
        """Update entity and invalidate cache."""
        await self._repository.update(entity)
        self._put_in_cache(entity)

    async def delete(self, id: str) -> None:
        """Delete entity and remove from cache."""
        await self._repository.delete(id)
        self._cache.pop(id, None)

    async def list(
        self,
        specification: Optional[BaseSpecification[T]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[T]:
        """List entities (bypasses cache)."""
        entities = await self._repository.list(specification, limit, offset)
        for entity in entities:
            self._put_in_cache(entity)
        return entities

    async def count(
        self,
        specification: Optional[BaseSpecification[T]] = None,
    ) -> int:
        """Count entities (bypasses cache)."""
        return await self._repository.count(specification)

    async def exists(self, id: str) -> bool:
        """Check if entity exists."""
        if self._is_cached(id):
            return True
        return await self._repository.exists(id)

    def invalidate(self, id: str) -> None:
        """Invalidate a cached entity."""
        self._cache.pop(id, None)

    def clear_cache(self) -> None:
        """Clear the entire cache."""
        self._cache.clear()


# =============================================================================
# Read-Only Repository
# =============================================================================


class ReadOnlyRepository(Generic[T]):
    """Read-only repository wrapper.

    Provides read-only access to a repository, useful for
    query-side of CQRS pattern.
    """

    def __init__(self, repository: Repository[T]):
        """Initialize read-only repository.

        Args:
            repository: Underlying repository
        """
        self._repository = repository

    async def get(self, id: str) -> Optional[T]:
        """Get entity by id."""
        return await self._repository.get(id)

    async def get_or_raise(self, id: str) -> T:
        """Get entity or raise exception."""
        return await self._repository.get_or_raise(id)

    async def list(
        self,
        specification: Optional[BaseSpecification[T]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[T]:
        """List entities matching specification."""
        return await self._repository.list(specification, limit, offset)

    async def count(
        self,
        specification: Optional[BaseSpecification[T]] = None,
    ) -> int:
        """Count entities matching specification."""
        return await self._repository.count(specification)

    async def exists(self, id: str) -> bool:
        """Check if entity exists."""
        return await self._repository.exists(id)


# =============================================================================
# Factory Functions
# =============================================================================


def create_repository(
    entity_class: Type[T],
    backend: str = "memory",
    **kwargs: Any,
) -> Repository[T]:
    """Factory function to create repositories.

    Args:
        entity_class: Entity class for the repository
        backend: Storage backend ("memory", "sqlite")
        **kwargs: Backend-specific arguments

    Returns:
        Repository instance

    Example:
        repo = create_repository(User, "sqlite", db_path="users.db", table_name="users")
    """
    if backend == "memory":
        return InMemoryRepository[T](entity_class)
    elif backend == "sqlite":
        return SQLiteRepository[T](
            db_path=kwargs.get("db_path", ":memory:"),
            table_name=kwargs.get("table_name", entity_class.__name__.lower()),
            entity_class=entity_class,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


def create_cached_repository(
    repository: Repository[T],
    cache_ttl: int = 300,
    max_cache_size: int = 1000,
) -> CachedRepository[T]:
    """Create a cached wrapper around a repository.

    Args:
        repository: Repository to cache
        cache_ttl: Cache TTL in seconds
        max_cache_size: Maximum cache size

    Returns:
        Cached repository
    """
    return CachedRepository[T](repository, cache_ttl, max_cache_size)
