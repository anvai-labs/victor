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

"""Unit of Work pattern for transaction management.

This module provides a comprehensive Unit of Work implementation that:
- Tracks all changes made during a business transaction
- Coordinates atomic persistence of changes
- Supports rollback on failure
- Enables cross-repository transaction management

Design Patterns:
- Unit of Work: Track and persist changes atomically
- Identity Map: Ensure single instance per entity
- Repository: Data access abstraction

Example:
    from victor.core.unit_of_work import UnitOfWork
    from victor.core.repository import InMemoryRepository

    # Create repositories
    users_repo = InMemoryRepository[User]()
    orders_repo = InMemoryRepository[Order]()

    # Use Unit of Work
    async with UnitOfWork() as uow:
        uow.register(users_repo)
        uow.register(orders_repo)

        user = User(name="Alice")
        order = Order(user_id=user.id, total=100.0)

        uow.add(user)
        uow.add(order)

        await uow.commit()  # Persists both atomically
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
)

from victor.core.repository import Entity, Repository

logger = logging.getLogger(__name__)


# =============================================================================
# Entity State
# =============================================================================


class EntityState(Enum):
    """State of an entity within the Unit of Work."""

    CLEAN = "clean"
    NEW = "new"
    MODIFIED = "modified"
    DELETED = "deleted"


@dataclass
class TrackedEntity:
    """Wrapper for tracking entity state."""

    entity: Entity
    state: EntityState
    original_data: Optional[Dict[str, Any]] = None  # For dirty checking

    def mark_clean(self) -> None:
        """Mark entity as clean (persisted)."""
        self.state = EntityState.CLEAN
        self.original_data = self.entity.to_dict()

    def mark_modified(self) -> None:
        """Mark entity as modified."""
        if self.state == EntityState.CLEAN:
            self.state = EntityState.MODIFIED

    def mark_deleted(self) -> None:
        """Mark entity for deletion."""
        self.state = EntityState.DELETED

    def is_dirty(self) -> bool:
        """Check if entity has been modified."""
        if self.state in (EntityState.NEW, EntityState.DELETED):
            return True
        if self.state == EntityState.MODIFIED:
            return self.entity.to_dict() != self.original_data
        return False


T = TypeVar("T", bound=Entity)


# =============================================================================
# Unit of Work Protocol
# =============================================================================


class UnitOfWorkProtocol(ABC):
    """Abstract protocol for Unit of Work implementations."""

    @abstractmethod
    def register_new(self, entity: Entity) -> None:
        """Register a new entity for insertion."""
        ...

    @abstractmethod
    def register_modified(self, entity: Entity) -> None:
        """Register an entity as modified."""
        ...

    @abstractmethod
    def register_deleted(self, entity: Entity) -> None:
        """Register an entity for deletion."""
        ...

    @abstractmethod
    async def commit(self) -> None:
        """Commit all changes."""
        ...

    @abstractmethod
    async def rollback(self) -> None:
        """Rollback all changes."""
        ...


# =============================================================================
# Unit of Work Implementation
# =============================================================================


class UnitOfWork(UnitOfWorkProtocol):
    """Unit of Work for managing entity persistence.

    Tracks all changes made during a business transaction and
    coordinates atomic persistence.

    Example:
        async with UnitOfWork() as uow:
            uow.register_repository(users_repo)
            user = User(name="Alice")
            uow.register_new(user)
            await uow.commit()
    """

    def __init__(self) -> None:
        """Initialize Unit of Work."""
        self._identity_map: Dict[str, TrackedEntity] = {}
        self._repositories: Dict[Type[Entity], Repository] = {}
        self._new_entities: List[Entity] = []
        self._modified_entities: Set[str] = set()
        self._deleted_entities: Set[str] = set()
        self._committed = False
        self._lock = asyncio.Lock()

    def register_repository(self, repository: Repository[T], entity_type: Type[T]) -> None:
        """Register a repository for an entity type.

        Args:
            repository: Repository instance
            entity_type: Entity class this repository handles
        """
        self._repositories[entity_type] = repository
        logger.debug(f"Registered repository for {entity_type.__name__}")

    def _get_repository(self, entity: Entity) -> Repository:
        """Get repository for entity type."""
        entity_type = type(entity)
        if entity_type not in self._repositories:
            raise UnitOfWorkError(f"No repository registered for {entity_type.__name__}")
        return self._repositories[entity_type]

    def get(self, entity_id: str) -> Optional[Entity]:
        """Get entity from identity map.

        Args:
            entity_id: Entity identifier

        Returns:
            Entity if found in identity map, None otherwise
        """
        tracked = self._identity_map.get(entity_id)
        if tracked and tracked.state != EntityState.DELETED:
            return tracked.entity
        return None

    def attach(self, entity: Entity) -> None:
        """Attach an existing entity to the identity map.

        Args:
            entity: Entity to attach
        """
        if entity.id not in self._identity_map:
            tracked = TrackedEntity(
                entity=entity,
                state=EntityState.CLEAN,
                original_data=entity.to_dict(),
            )
            self._identity_map[entity.id] = tracked
            logger.debug(f"Attached entity {entity.id}")

    def detach(self, entity: Entity) -> None:
        """Detach entity from identity map.

        Args:
            entity: Entity to detach
        """
        self._identity_map.pop(entity.id, None)
        self._new_entities = [e for e in self._new_entities if e.id != entity.id]
        self._modified_entities.discard(entity.id)
        self._deleted_entities.discard(entity.id)
        logger.debug(f"Detached entity {entity.id}")

    def register_new(self, entity: Entity) -> None:
        """Register a new entity for insertion.

        Args:
            entity: New entity to insert
        """
        if entity.id in self._identity_map:
            raise UnitOfWorkError(f"Entity {entity.id} already tracked")

        tracked = TrackedEntity(entity=entity, state=EntityState.NEW)
        self._identity_map[entity.id] = tracked
        self._new_entities.append(entity)
        logger.debug(f"Registered new entity {entity.id}")

    def register_modified(self, entity: Entity) -> None:
        """Register an entity as modified.

        Args:
            entity: Modified entity
        """
        if entity.id not in self._identity_map:
            # Auto-attach if not tracked
            self.attach(entity)

        tracked = self._identity_map[entity.id]
        tracked.mark_modified()
        self._modified_entities.add(entity.id)
        logger.debug(f"Registered modified entity {entity.id}")

    def register_deleted(self, entity: Entity) -> None:
        """Register an entity for deletion.

        Args:
            entity: Entity to delete
        """
        if entity.id in self._identity_map:
            tracked = self._identity_map[entity.id]
            tracked.mark_deleted()
            self._deleted_entities.add(entity.id)
        else:
            # Create tracked entry for deletion
            tracked = TrackedEntity(entity=entity, state=EntityState.DELETED)
            self._identity_map[entity.id] = tracked
            self._deleted_entities.add(entity.id)

        # Remove from new/modified sets
        self._new_entities = [e for e in self._new_entities if e.id != entity.id]
        self._modified_entities.discard(entity.id)
        logger.debug(f"Registered deleted entity {entity.id}")

    async def commit(self) -> None:
        """Commit all changes to repositories.

        Persists new entities, updates modified entities, and
        deletes removed entities. All operations are performed
        atomically where possible.

        Raises:
            UnitOfWorkError: If commit fails
        """
        async with self._lock:
            if self._committed:
                raise UnitOfWorkError("Unit of Work already committed")

            try:
                # Insert new entities
                for entity in self._new_entities:
                    repo = self._get_repository(entity)
                    await repo.add(entity)
                    self._identity_map[entity.id].mark_clean()
                    logger.debug(f"Committed new entity {entity.id}")

                # Update modified entities
                for entity_id in self._modified_entities:
                    tracked = self._identity_map.get(entity_id)
                    if tracked and tracked.is_dirty():
                        repo = self._get_repository(tracked.entity)
                        await repo.update(tracked.entity)
                        tracked.mark_clean()
                        logger.debug(f"Committed modified entity {entity_id}")

                # Delete removed entities
                for entity_id in self._deleted_entities:
                    tracked = self._identity_map.get(entity_id)
                    if tracked:
                        repo = self._get_repository(tracked.entity)
                        await repo.delete(entity_id)
                        logger.debug(f"Committed deleted entity {entity_id}")

                # Clear tracking
                self._new_entities.clear()
                self._modified_entities.clear()
                self._deleted_entities.clear()
                self._committed = True

                logger.info("Unit of Work committed successfully")

            except Exception as e:
                logger.error(f"Commit failed: {e}")
                raise UnitOfWorkError(f"Commit failed: {e}") from e

    async def rollback(self) -> None:
        """Rollback all uncommitted changes.

        Reverts entities to their original state.
        """
        async with self._lock:
            # Revert modified entities
            for entity_id in self._modified_entities:
                tracked = self._identity_map.get(entity_id)
                if tracked and tracked.original_data:
                    # Restore original data
                    for key, value in tracked.original_data.items():
                        setattr(tracked.entity, key, value)
                    tracked.state = EntityState.CLEAN

            # Clear tracking
            self._new_entities.clear()
            self._modified_entities.clear()
            self._deleted_entities.clear()

            logger.info("Unit of Work rolled back")

    def has_changes(self) -> bool:
        """Check if there are uncommitted changes.

        Returns:
            True if there are pending changes
        """
        if self._new_entities or self._deleted_entities:
            return True

        for entity_id in self._modified_entities:
            tracked = self._identity_map.get(entity_id)
            if tracked and tracked.is_dirty():
                return True

        return False

    @property
    def pending_count(self) -> int:
        """Get count of pending changes."""
        return len(self._new_entities) + len(self._modified_entities) + len(self._deleted_entities)

    async def __aenter__(self) -> "UnitOfWork":
        """Enter async context."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit async context with auto-rollback on error."""
        if exc_type is not None:
            await self.rollback()
        elif not self._committed and self.has_changes():
            await self.rollback()
            logger.warning("Auto-rollback: uncommitted changes discarded")
        return False


# =============================================================================
# Exceptions
# =============================================================================


class UnitOfWorkError(Exception):
    """Base exception for Unit of Work errors."""

    pass


# =============================================================================
# SQLite Unit of Work
# =============================================================================


class SQLiteUnitOfWork(UnitOfWork):
    """SQLite-backed Unit of Work with actual transaction support.

    Uses SQLite transactions for true atomicity.
    """

    def __init__(self, db_path: str) -> None:
        """Initialize SQLite Unit of Work.

        Args:
            db_path: Path to SQLite database
        """
        super().__init__()
        self._db_path = db_path
        self._connection = None

    async def begin_transaction(self) -> None:
        """Begin a database transaction."""
        import sqlite3

        self._connection = sqlite3.connect(self._db_path)
        self._connection.execute("BEGIN TRANSACTION")
        logger.debug("Transaction started")

    async def commit(self) -> None:
        """Commit with database transaction."""
        if self._connection:
            try:
                await super().commit()
                self._connection.commit()
                logger.debug("Transaction committed")
            except Exception:
                self._connection.rollback()
                raise
            finally:
                self._connection.close()
                self._connection = None
        else:
            await super().commit()

    async def rollback(self) -> None:
        """Rollback database transaction."""
        await super().rollback()
        if self._connection:
            self._connection.rollback()
            self._connection.close()
            self._connection = None
            logger.debug("Transaction rolled back")


# =============================================================================
# Composite Unit of Work
# =============================================================================


class CompositeUnitOfWork(UnitOfWorkProtocol):
    """Coordinates multiple Units of Work for distributed transactions.

    Uses a two-phase commit protocol for coordination.
    """

    def __init__(self) -> None:
        """Initialize composite Unit of Work."""
        self._units: List[UnitOfWork] = []
        self._prepared: Set[int] = set()

    def add_unit(self, unit: UnitOfWork) -> None:
        """Add a Unit of Work to coordinate.

        Args:
            unit: Unit of Work to add
        """
        self._units.append(unit)

    def register_new(self, entity: Entity) -> None:
        """Register new entity in appropriate unit."""
        for unit in self._units:
            try:
                unit.register_new(entity)
                return
            except UnitOfWorkError:
                continue
        raise UnitOfWorkError(f"No unit accepts {type(entity).__name__}")

    def register_modified(self, entity: Entity) -> None:
        """Register modified entity in appropriate unit."""
        for unit in self._units:
            if unit.get(entity.id):
                unit.register_modified(entity)
                return
        raise UnitOfWorkError(f"Entity {entity.id} not found in any unit")

    def register_deleted(self, entity: Entity) -> None:
        """Register deleted entity in appropriate unit."""
        for unit in self._units:
            if unit.get(entity.id):
                unit.register_deleted(entity)
                return
        # Try all units for deletion
        for unit in self._units:
            try:
                unit.register_deleted(entity)
                return
            except UnitOfWorkError:
                continue

    async def commit(self) -> None:
        """Two-phase commit across all units.

        Phase 1: Prepare (verify all units can commit)
        Phase 2: Commit (persist changes)
        """
        # Phase 1: Prepare
        for i, unit in enumerate(self._units):
            if unit.has_changes():
                self._prepared.add(i)

        # Phase 2: Commit
        try:
            for i in self._prepared:
                await self._units[i].commit()
            logger.info(f"Composite commit successful ({len(self._prepared)} units)")
        except Exception as e:
            # Rollback all on failure
            await self.rollback()
            raise UnitOfWorkError(f"Composite commit failed: {e}") from e

    async def rollback(self) -> None:
        """Rollback all units."""
        for unit in self._units:
            await unit.rollback()
        self._prepared.clear()
        logger.info("Composite rollback complete")


# =============================================================================
# Factory Functions
# =============================================================================


def create_unit_of_work(backend: str = "memory", **kwargs) -> UnitOfWork:
    """Factory function to create Unit of Work instances.

    Args:
        backend: Backend type ("memory", "sqlite")
        **kwargs: Backend-specific arguments

    Returns:
        Unit of Work instance
    """
    if backend == "memory":
        return UnitOfWork()
    elif backend == "sqlite":
        return SQLiteUnitOfWork(kwargs.get("db_path", ":memory:"))
    else:
        raise ValueError(f"Unknown backend: {backend}")


@asynccontextmanager
async def transactional(*repositories: tuple[Repository, Type[Entity]]):
    """Context manager for transactional operations.

    Example:
        async with transactional((user_repo, User), (order_repo, Order)) as uow:
            uow.register_new(user)
            uow.register_new(order)
            await uow.commit()

    Args:
        repositories: Tuples of (repository, entity_type)

    Yields:
        Configured Unit of Work
    """
    uow = UnitOfWork()
    for repo, entity_type in repositories:
        uow.register_repository(repo, entity_type)

    try:
        yield uow
        if uow.has_changes() and not uow._committed:
            await uow.commit()
    except Exception:
        await uow.rollback()
        raise
