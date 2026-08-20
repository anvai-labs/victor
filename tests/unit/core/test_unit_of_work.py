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

"""Tests for Unit of Work pattern module."""

from dataclasses import dataclass

import pytest

from victor.core.repository import Entity, InMemoryRepository
from victor.core.unit_of_work import (
    CompositeUnitOfWork,
    EntityState,
    TrackedEntity,
    UnitOfWork,
    UnitOfWorkError,
    create_unit_of_work,
    transactional,
)

# =============================================================================
# Test Entities
# =============================================================================


@dataclass
class User(Entity):
    """Test user entity."""

    name: str = ""
    email: str = ""


@dataclass
class Order(Entity):
    """Test order entity."""

    user_id: str = ""
    total: float = 0.0
    status: str = "pending"


# =============================================================================
# TrackedEntity Tests
# =============================================================================


class TestTrackedEntity:
    """Tests for TrackedEntity."""

    def test_creation(self):
        """Test creating tracked entity."""
        user = User(name="Alice")
        tracked = TrackedEntity(entity=user, state=EntityState.NEW)

        assert tracked.entity == user
        assert tracked.state == EntityState.NEW

    def test_mark_clean(self):
        """Test marking entity as clean."""
        user = User(name="Alice")
        tracked = TrackedEntity(entity=user, state=EntityState.NEW)

        tracked.mark_clean()

        assert tracked.state == EntityState.CLEAN
        assert tracked.original_data is not None

    def test_mark_modified(self):
        """Test marking entity as modified."""
        user = User(name="Alice")
        tracked = TrackedEntity(entity=user, state=EntityState.CLEAN)

        tracked.mark_modified()

        assert tracked.state == EntityState.MODIFIED

    def test_mark_modified_from_new_stays_new(self):
        """Test new entity stays new when modified."""
        user = User(name="Alice")
        tracked = TrackedEntity(entity=user, state=EntityState.NEW)

        tracked.mark_modified()

        assert tracked.state == EntityState.NEW  # Still new

    def test_mark_deleted(self):
        """Test marking entity as deleted."""
        user = User(name="Alice")
        tracked = TrackedEntity(entity=user, state=EntityState.CLEAN)

        tracked.mark_deleted()

        assert tracked.state == EntityState.DELETED

    def test_is_dirty_new(self):
        """Test new entity is dirty."""
        user = User(name="Alice")
        tracked = TrackedEntity(entity=user, state=EntityState.NEW)

        assert tracked.is_dirty() is True

    def test_is_dirty_clean(self):
        """Test clean entity is not dirty."""
        user = User(name="Alice")
        tracked = TrackedEntity(entity=user, state=EntityState.CLEAN, original_data=user.to_dict())

        assert tracked.is_dirty() is False

    def test_is_dirty_modified(self):
        """Test modified entity is dirty."""
        user = User(name="Alice")
        tracked = TrackedEntity(
            entity=user, state=EntityState.MODIFIED, original_data={"name": "Bob"}
        )

        assert tracked.is_dirty() is True


# =============================================================================
# UnitOfWork Basic Tests
# =============================================================================


class TestUnitOfWork:
    """Tests for UnitOfWork."""

    @pytest.mark.asyncio
    async def test_register_repository(self):
        """Test registering a repository."""
        uow = UnitOfWork()
        repo = InMemoryRepository[User]()

        uow.register_repository(repo, User)

        # Should not raise
        user = User(name="Alice")
        uow.register_new(user)

    @pytest.mark.asyncio
    async def test_register_new(self):
        """Test registering new entity."""
        uow = UnitOfWork()
        repo = InMemoryRepository[User]()
        uow.register_repository(repo, User)

        user = User(id="user-1", name="Alice")
        uow.register_new(user)

        assert uow.pending_count == 1
        assert uow.get("user-1") == user

    @pytest.mark.asyncio
    async def test_register_new_duplicate_raises(self):
        """Test registering duplicate entity raises."""
        uow = UnitOfWork()
        repo = InMemoryRepository[User]()
        uow.register_repository(repo, User)

        user = User(id="user-1", name="Alice")
        uow.register_new(user)

        with pytest.raises(UnitOfWorkError):
            uow.register_new(User(id="user-1", name="Bob"))

    @pytest.mark.asyncio
    async def test_register_modified(self):
        """Test registering modified entity."""
        uow = UnitOfWork()
        repo = InMemoryRepository[User]()
        uow.register_repository(repo, User)

        user = User(id="user-1", name="Alice")
        uow.attach(user)

        user.name = "Alice Updated"
        uow.register_modified(user)

        assert "user-1" in uow._modified_entities

    @pytest.mark.asyncio
    async def test_register_deleted(self):
        """Test registering deleted entity."""
        uow = UnitOfWork()
        repo = InMemoryRepository[User]()
        uow.register_repository(repo, User)

        user = User(id="user-1", name="Alice")
        uow.attach(user)

        uow.register_deleted(user)

        assert "user-1" in uow._deleted_entities

    @pytest.mark.asyncio
    async def test_attach_and_get(self):
        """Test attaching and getting entity."""
        uow = UnitOfWork()
        user = User(id="user-1", name="Alice")

        uow.attach(user)

        assert uow.get("user-1") == user

    @pytest.mark.asyncio
    async def test_detach(self):
        """Test detaching entity."""
        uow = UnitOfWork()
        user = User(id="user-1", name="Alice")
        uow.attach(user)

        uow.detach(user)

        assert uow.get("user-1") is None


# =============================================================================
# UnitOfWork Commit Tests
# =============================================================================


class TestUnitOfWorkCommit:
    """Tests for UnitOfWork commit functionality."""

    @pytest.mark.asyncio
    async def test_commit_new_entity(self):
        """Test committing new entity."""
        uow = UnitOfWork()
        repo = InMemoryRepository[User]()
        uow.register_repository(repo, User)

        user = User(id="user-1", name="Alice")
        uow.register_new(user)

        await uow.commit()

        # Verify persisted
        saved = await repo.get("user-1")
        assert saved is not None
        assert saved.name == "Alice"

    @pytest.mark.asyncio
    async def test_commit_modified_entity(self):
        """Test committing modified entity."""
        repo = InMemoryRepository[User]()
        user = User(id="user-1", name="Alice")
        await repo.add(user)

        uow = UnitOfWork()
        uow.register_repository(repo, User)
        uow.attach(user)

        user.name = "Alice Updated"
        uow.register_modified(user)

        await uow.commit()

        # Verify updated
        saved = await repo.get("user-1")
        assert saved.name == "Alice Updated"

    @pytest.mark.asyncio
    async def test_commit_deleted_entity(self):
        """Test committing deleted entity."""
        repo = InMemoryRepository[User]()
        user = User(id="user-1", name="Alice")
        await repo.add(user)

        uow = UnitOfWork()
        uow.register_repository(repo, User)
        uow.attach(user)
        uow.register_deleted(user)

        await uow.commit()

        # Verify deleted
        saved = await repo.get("user-1")
        assert saved is None

    @pytest.mark.asyncio
    async def test_commit_multiple_entities(self):
        """Test committing multiple entities."""
        user_repo = InMemoryRepository[User]()
        order_repo = InMemoryRepository[Order]()

        uow = UnitOfWork()
        uow.register_repository(user_repo, User)
        uow.register_repository(order_repo, Order)

        user = User(id="user-1", name="Alice")
        order = Order(id="order-1", user_id="user-1", total=100.0)

        uow.register_new(user)
        uow.register_new(order)

        await uow.commit()

        # Verify both persisted
        assert await user_repo.get("user-1") is not None
        assert await order_repo.get("order-1") is not None

    @pytest.mark.asyncio
    async def test_commit_twice_raises(self):
        """Test committing twice raises error."""
        uow = UnitOfWork()
        repo = InMemoryRepository[User]()
        uow.register_repository(repo, User)

        user = User(name="Alice")
        uow.register_new(user)

        await uow.commit()

        with pytest.raises(UnitOfWorkError):
            await uow.commit()


# =============================================================================
# UnitOfWork Rollback Tests
# =============================================================================


class TestUnitOfWorkRollback:
    """Tests for UnitOfWork rollback functionality."""

    @pytest.mark.asyncio
    async def test_rollback_reverts_changes(self):
        """Test rollback reverts modified entity."""
        uow = UnitOfWork()
        user = User(id="user-1", name="Alice")
        uow.attach(user)

        user.name = "Modified"
        uow.register_modified(user)

        await uow.rollback()

        assert user.name == "Alice"  # Reverted

    @pytest.mark.asyncio
    async def test_rollback_clears_pending(self):
        """Test rollback clears pending changes."""
        uow = UnitOfWork()
        repo = InMemoryRepository[User]()
        uow.register_repository(repo, User)

        user = User(name="Alice")
        uow.register_new(user)

        await uow.rollback()

        assert not uow.has_changes()


# =============================================================================
# UnitOfWork Context Manager Tests
# =============================================================================


class TestUnitOfWorkContext:
    """Tests for UnitOfWork context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_commits_on_success(self):
        """Test context manager auto-commits."""
        repo = InMemoryRepository[User]()

        async with UnitOfWork() as uow:
            uow.register_repository(repo, User)
            user = User(id="user-1", name="Alice")
            uow.register_new(user)
            await uow.commit()

        assert await repo.get("user-1") is not None

    @pytest.mark.asyncio
    async def test_context_manager_rollback_on_error(self):
        """Test context manager rolls back on error."""
        repo = InMemoryRepository[User]()

        try:
            async with UnitOfWork() as uow:
                uow.register_repository(repo, User)
                user = User(id="user-1", name="Alice")
                uow.register_new(user)
                raise ValueError("Test error")
        except ValueError:
            pass

        # Not committed due to error
        assert await repo.get("user-1") is None

    @pytest.mark.asyncio
    async def test_context_manager_rollback_uncommitted(self):
        """Test uncommitted changes are rolled back."""
        repo = InMemoryRepository[User]()

        async with UnitOfWork() as uow:
            uow.register_repository(repo, User)
            user = User(id="user-1", name="Alice")
            uow.register_new(user)
            # Not committing

        # Should be rolled back (not persisted)
        assert await repo.get("user-1") is None


# =============================================================================
# CompositeUnitOfWork Tests
# =============================================================================


class TestCompositeUnitOfWork:
    """Tests for CompositeUnitOfWork."""

    @pytest.mark.asyncio
    async def test_composite_commit(self):
        """Test composite commit across units."""
        user_repo = InMemoryRepository[User]()
        order_repo = InMemoryRepository[Order]()

        uow1 = UnitOfWork()
        uow1.register_repository(user_repo, User)

        uow2 = UnitOfWork()
        uow2.register_repository(order_repo, Order)

        composite = CompositeUnitOfWork()
        composite.add_unit(uow1)
        composite.add_unit(uow2)

        user = User(id="user-1", name="Alice")
        order = Order(id="order-1", user_id="user-1", total=100.0)

        uow1.register_new(user)
        uow2.register_new(order)

        await composite.commit()

        assert await user_repo.get("user-1") is not None
        assert await order_repo.get("order-1") is not None

    @pytest.mark.asyncio
    async def test_composite_rollback(self):
        """Test composite rollback."""
        user_repo = InMemoryRepository[User]()

        uow1 = UnitOfWork()
        uow1.register_repository(user_repo, User)

        composite = CompositeUnitOfWork()
        composite.add_unit(uow1)

        user = User(id="user-1", name="Alice")
        uow1.register_new(user)

        await composite.rollback()

        assert not uow1.has_changes()


# =============================================================================
# Factory Tests
# =============================================================================


class TestFactory:
    """Tests for factory functions."""

    def test_create_memory_unit_of_work(self):
        """Test creating memory Unit of Work."""
        uow = create_unit_of_work("memory")

        assert isinstance(uow, UnitOfWork)

    def test_create_unknown_backend_raises(self):
        """Test unknown backend raises error."""
        with pytest.raises(ValueError):
            create_unit_of_work("unknown")


# =============================================================================
# Transactional Context Tests
# =============================================================================


class TestTransactional:
    """Tests for transactional context manager."""

    @pytest.mark.asyncio
    async def test_transactional_basic(self):
        """Test basic transactional context."""
        user_repo = InMemoryRepository[User]()

        async with transactional((user_repo, User)) as uow:
            user = User(id="user-1", name="Alice")
            uow.register_new(user)

        assert await user_repo.get("user-1") is not None

    @pytest.mark.asyncio
    async def test_transactional_multiple_repos(self):
        """Test transactional with multiple repos."""
        user_repo = InMemoryRepository[User]()
        order_repo = InMemoryRepository[Order]()

        async with transactional((user_repo, User), (order_repo, Order)) as uow:
            user = User(id="user-1", name="Alice")
            order = Order(id="order-1", user_id="user-1")
            uow.register_new(user)
            uow.register_new(order)

        assert await user_repo.get("user-1") is not None
        assert await order_repo.get("order-1") is not None

    @pytest.mark.asyncio
    async def test_transactional_rollback_on_error(self):
        """Test transactional rolls back on error."""
        user_repo = InMemoryRepository[User]()

        try:
            async with transactional((user_repo, User)) as uow:
                user = User(id="user-1", name="Alice")
                uow.register_new(user)
                raise ValueError("Test error")
        except ValueError:
            pass

        assert await user_repo.get("user-1") is None
