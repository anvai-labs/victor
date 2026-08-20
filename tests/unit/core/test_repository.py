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

"""Tests for repository pattern module."""

import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pytest

from victor.core.repository import (
    AndSpecification,
    AttributeSpecification,
    BaseSpecification,
    CachedRepository,
    ConcurrencyError,
    Entity,
    EntityExistsError,
    EntityNotFoundError,
    InMemoryRepository,
    NotSpecification,
    OrSpecification,
    ReadOnlyRepository,
    Repository,
    SQLiteRepository,
    create_cached_repository,
    create_repository,
)

# =============================================================================
# Test Entities
# =============================================================================


@dataclass
class User(Entity):
    """Test user entity."""

    name: str = ""
    email: str = ""
    age: int = 0
    active: bool = True


@dataclass
class Product(Entity):
    """Test product entity."""

    name: str = ""
    price: float = 0.0
    category: str = ""


# =============================================================================
# Entity Tests
# =============================================================================


class TestEntity:
    """Tests for Entity base class."""

    def test_entity_creation(self):
        """Test creating an entity."""
        entity = Entity()

        assert entity.id is not None
        assert entity.created_at is not None
        assert entity.updated_at is not None
        assert entity.version == 1

    def test_entity_with_custom_id(self):
        """Test creating entity with custom id."""
        entity = Entity(id="custom-123")

        assert entity.id == "custom-123"

    def test_entity_equality(self):
        """Test entities are equal by id."""
        e1 = Entity(id="same-id")
        e2 = Entity(id="same-id")
        e3 = Entity(id="different-id")

        assert e1 == e2
        assert e1 != e3

    def test_entity_hash(self):
        """Test entity hashing."""
        e1 = Entity(id="same-id")
        e2 = Entity(id="same-id")

        assert hash(e1) == hash(e2)

        # Can be used in sets
        entities = {e1, e2}
        assert len(entities) == 1

    def test_to_dict(self):
        """Test entity serialization."""
        entity = Entity(id="test-1")
        data = entity.to_dict()

        assert data["id"] == "test-1"
        assert "created_at" in data
        assert "updated_at" in data
        assert data["version"] == 1

    def test_from_dict(self):
        """Test entity deserialization."""
        now = datetime.now(timezone.utc)
        data = {
            "id": "test-1",
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "version": 2,
        }

        entity = Entity.from_dict(data)

        assert entity.id == "test-1"
        assert entity.version == 2

    def test_touch(self):
        """Test touch updates updated_at."""
        entity = Entity()
        original = entity.updated_at

        import time

        time.sleep(0.01)
        entity.touch()

        assert entity.updated_at > original

    def test_increment_version(self):
        """Test version increment."""
        entity = Entity()
        assert entity.version == 1

        entity.increment_version()
        assert entity.version == 2


# =============================================================================
# Specification Tests
# =============================================================================


class TestSpecification:
    """Tests for specification pattern."""

    def test_attribute_specification_eq(self):
        """Test equality specification."""
        spec = AttributeSpecification[User]("name", "Alice")
        user1 = User(name="Alice")
        user2 = User(name="Bob")

        assert spec.is_satisfied_by(user1) is True
        assert spec.is_satisfied_by(user2) is False

    def test_attribute_specification_ne(self):
        """Test not equal specification."""
        spec = AttributeSpecification[User]("name", "Alice", "ne")
        user1 = User(name="Alice")
        user2 = User(name="Bob")

        assert spec.is_satisfied_by(user1) is False
        assert spec.is_satisfied_by(user2) is True

    def test_attribute_specification_gt(self):
        """Test greater than specification."""
        spec = AttributeSpecification[User]("age", 18, "gt")
        user1 = User(age=25)
        user2 = User(age=15)

        assert spec.is_satisfied_by(user1) is True
        assert spec.is_satisfied_by(user2) is False

    def test_attribute_specification_in(self):
        """Test in specification."""
        spec = AttributeSpecification[User]("name", ["Alice", "Bob"], "in")
        user1 = User(name="Alice")
        user2 = User(name="Charlie")

        assert spec.is_satisfied_by(user1) is True
        assert spec.is_satisfied_by(user2) is False

    def test_attribute_specification_contains(self):
        """Test contains specification."""
        spec = AttributeSpecification[User]("email", "@example.com", "contains")
        user1 = User(email="alice@example.com")
        user2 = User(email="alice@other.com")

        assert spec.is_satisfied_by(user1) is True
        assert spec.is_satisfied_by(user2) is False

    def test_and_specification(self):
        """Test AND combination."""
        spec1 = AttributeSpecification[User]("active", True)
        spec2 = AttributeSpecification[User]("age", 18, "gte")

        combined = spec1 & spec2

        user1 = User(active=True, age=25)
        user2 = User(active=False, age=25)
        user3 = User(active=True, age=15)

        assert combined.is_satisfied_by(user1) is True
        assert combined.is_satisfied_by(user2) is False
        assert combined.is_satisfied_by(user3) is False

    def test_or_specification(self):
        """Test OR combination."""
        spec1 = AttributeSpecification[User]("name", "Alice")
        spec2 = AttributeSpecification[User]("name", "Bob")

        combined = spec1 | spec2

        user1 = User(name="Alice")
        user2 = User(name="Bob")
        user3 = User(name="Charlie")

        assert combined.is_satisfied_by(user1) is True
        assert combined.is_satisfied_by(user2) is True
        assert combined.is_satisfied_by(user3) is False

    def test_not_specification(self):
        """Test NOT specification."""
        spec = AttributeSpecification[User]("active", True)
        negated = ~spec

        user1 = User(active=True)
        user2 = User(active=False)

        assert negated.is_satisfied_by(user1) is False
        assert negated.is_satisfied_by(user2) is True

    def test_complex_specification(self):
        """Test complex specification combination."""
        # (active AND age >= 18) OR (name == "Admin")
        active_adult = AttributeSpecification[User]("active", True) & AttributeSpecification[User](
            "age", 18, "gte"
        )
        admin = AttributeSpecification[User]("name", "Admin")
        combined = active_adult | admin

        user1 = User(name="Alice", active=True, age=25)  # matches
        user2 = User(name="Admin", active=False, age=10)  # matches (admin)
        user3 = User(name="Bob", active=False, age=25)  # no match

        assert combined.is_satisfied_by(user1) is True
        assert combined.is_satisfied_by(user2) is True
        assert combined.is_satisfied_by(user3) is False

    def test_to_query(self):
        """Test specification to query conversion."""
        spec = AttributeSpecification[User]("age", 18, "gte")
        query = spec.to_query()

        assert query == {"age": {"$gte": 18}}


# =============================================================================
# InMemoryRepository Tests
# =============================================================================


class TestInMemoryRepository:
    """Tests for InMemoryRepository."""

    @pytest.mark.asyncio
    async def test_add_and_get(self):
        """Test adding and retrieving entity."""
        repo = InMemoryRepository[User]()
        user = User(id="user-1", name="Alice", email="alice@example.com")

        await repo.add(user)
        retrieved = await repo.get("user-1")

        assert retrieved is not None
        assert retrieved.id == "user-1"
        assert retrieved.name == "Alice"

    @pytest.mark.asyncio
    async def test_add_duplicate_raises(self):
        """Test adding duplicate entity raises error."""
        repo = InMemoryRepository[User]()
        user = User(id="user-1", name="Alice")

        await repo.add(user)

        with pytest.raises(EntityExistsError):
            await repo.add(User(id="user-1", name="Bob"))

    @pytest.mark.asyncio
    async def test_get_nonexistent(self):
        """Test getting nonexistent entity returns None."""
        repo = InMemoryRepository[User]()

        result = await repo.get("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_update(self):
        """Test updating entity."""
        repo = InMemoryRepository[User]()
        user = User(id="user-1", name="Alice")
        await repo.add(user)

        user.name = "Alice Smith"
        await repo.update(user)

        retrieved = await repo.get("user-1")
        assert retrieved.name == "Alice Smith"
        assert retrieved.version == 2

    @pytest.mark.asyncio
    async def test_update_nonexistent_raises(self):
        """Test updating nonexistent entity raises error."""
        repo = InMemoryRepository[User]()
        user = User(id="nonexistent", name="Alice")

        with pytest.raises(EntityNotFoundError):
            await repo.update(user)

    @pytest.mark.asyncio
    async def test_update_version_conflict(self):
        """Test version conflict detection."""
        repo = InMemoryRepository[User]()
        user1 = User(id="user-1", name="Alice")
        await repo.add(user1)

        # Get a reference and update it (this increments version to 2)
        user2 = await repo.get("user-1")
        user2.name = "Alice Updated"
        await repo.update(user2)  # version becomes 2

        # Create a new entity with same ID but old version (simulating stale data)
        stale_user = User(id="user-1", name="Alice Stale", version=1)
        with pytest.raises(ConcurrencyError):
            await repo.update(stale_user)  # has version 1, store has version 2

    @pytest.mark.asyncio
    async def test_delete(self):
        """Test deleting entity."""
        repo = InMemoryRepository[User]()
        user = User(id="user-1", name="Alice")
        await repo.add(user)

        await repo.delete("user-1")

        assert await repo.get("user-1") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_raises(self):
        """Test deleting nonexistent entity raises error."""
        repo = InMemoryRepository[User]()

        with pytest.raises(EntityNotFoundError):
            await repo.delete("nonexistent")

    @pytest.mark.asyncio
    async def test_list(self):
        """Test listing entities."""
        repo = InMemoryRepository[User]()
        await repo.add(User(id="1", name="Alice"))
        await repo.add(User(id="2", name="Bob"))
        await repo.add(User(id="3", name="Charlie"))

        users = await repo.list()

        assert len(users) == 3

    @pytest.mark.asyncio
    async def test_list_with_specification(self):
        """Test listing with filter."""
        repo = InMemoryRepository[User]()
        await repo.add(User(id="1", name="Alice", active=True))
        await repo.add(User(id="2", name="Bob", active=False))
        await repo.add(User(id="3", name="Charlie", active=True))

        spec = AttributeSpecification[User]("active", True)
        users = await repo.list(spec)

        assert len(users) == 2
        assert all(u.active for u in users)

    @pytest.mark.asyncio
    async def test_list_with_pagination(self):
        """Test listing with limit and offset."""
        repo = InMemoryRepository[User]()
        for i in range(10):
            await repo.add(User(id=str(i), name=f"User {i}"))

        page1 = await repo.list(limit=3, offset=0)
        page2 = await repo.list(limit=3, offset=3)

        assert len(page1) == 3
        assert len(page2) == 3

    @pytest.mark.asyncio
    async def test_count(self):
        """Test counting entities."""
        repo = InMemoryRepository[User]()
        await repo.add(User(id="1", name="Alice"))
        await repo.add(User(id="2", name="Bob"))

        count = await repo.count()

        assert count == 2

    @pytest.mark.asyncio
    async def test_count_with_specification(self):
        """Test counting with filter."""
        repo = InMemoryRepository[User]()
        await repo.add(User(id="1", active=True))
        await repo.add(User(id="2", active=False))
        await repo.add(User(id="3", active=True))

        spec = AttributeSpecification[User]("active", True)
        count = await repo.count(spec)

        assert count == 2

    @pytest.mark.asyncio
    async def test_exists(self):
        """Test existence check."""
        repo = InMemoryRepository[User]()
        await repo.add(User(id="user-1", name="Alice"))

        assert await repo.exists("user-1") is True
        assert await repo.exists("user-2") is False

    @pytest.mark.asyncio
    async def test_get_or_raise(self):
        """Test get_or_raise method."""
        repo = InMemoryRepository[User]()
        await repo.add(User(id="user-1", name="Alice"))

        user = await repo.get_or_raise("user-1")
        assert user.name == "Alice"

        with pytest.raises(EntityNotFoundError):
            await repo.get_or_raise("nonexistent")

    @pytest.mark.asyncio
    async def test_add_many(self):
        """Test adding multiple entities."""
        repo = InMemoryRepository[User]()
        users = [
            User(id="1", name="Alice"),
            User(id="2", name="Bob"),
            User(id="3", name="Charlie"),
        ]

        await repo.add_many(users)

        assert await repo.count() == 3

    @pytest.mark.asyncio
    async def test_delete_many(self):
        """Test deleting multiple entities."""
        repo = InMemoryRepository[User]()
        await repo.add_many([User(id=str(i)) for i in range(5)])

        await repo.delete_many(["0", "1", "2"])

        assert await repo.count() == 2


# =============================================================================
# SQLiteRepository Tests
# =============================================================================


class TestSQLiteRepository:
    """Tests for SQLiteRepository."""

    @pytest.mark.asyncio
    async def test_add_and_get(self):
        """Test adding and retrieving entity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            repo = SQLiteRepository[User](db_path, "users", User)

            user = User(id="user-1", name="Alice", email="alice@example.com")
            await repo.add(user)

            retrieved = await repo.get("user-1")
            assert retrieved is not None
            assert retrieved.name == "Alice"

    @pytest.mark.asyncio
    async def test_persistence(self):
        """Test data persists across instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            # Write with first instance
            repo1 = SQLiteRepository[User](db_path, "users", User)
            await repo1.add(User(id="user-1", name="Alice"))

            # Read with second instance
            repo2 = SQLiteRepository[User](db_path, "users", User)
            retrieved = await repo2.get("user-1")

            assert retrieved is not None
            assert retrieved.name == "Alice"

    @pytest.mark.asyncio
    async def test_update_with_optimistic_locking(self):
        """Test optimistic concurrency control."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            repo = SQLiteRepository[User](db_path, "users", User)

            user = User(id="user-1", name="Alice")
            await repo.add(user)

            # Update normally
            user.name = "Alice Updated"
            await repo.update(user)

            # Try to update with old version
            stale = User(id="user-1", name="Stale", version=1)
            with pytest.raises(ConcurrencyError):
                await repo.update(stale)

    @pytest.mark.asyncio
    async def test_list_with_specification(self):
        """Test listing with filter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            repo = SQLiteRepository[User](db_path, "users", User)

            await repo.add(User(id="1", name="Alice", active=True))
            await repo.add(User(id="2", name="Bob", active=False))

            spec = AttributeSpecification[User]("active", True)
            users = await repo.list(spec)

            assert len(users) == 1
            assert users[0].name == "Alice"

    @pytest.mark.asyncio
    async def test_count(self):
        """Test counting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            repo = SQLiteRepository[User](db_path, "users", User)

            await repo.add(User(id="1"))
            await repo.add(User(id="2"))

            assert await repo.count() == 2


# =============================================================================
# CachedRepository Tests
# =============================================================================


class TestCachedRepository:
    """Tests for CachedRepository."""

    @pytest.mark.asyncio
    async def test_caching_get(self):
        """Test get uses cache and avoids redundant lookups."""
        # Track how many times the underlying get is called
        call_count = 0
        inner = InMemoryRepository[User]()

        # Add user directly to inner repo (not through cache)
        user = User(id="user-1", name="Alice")
        await inner.add(user)

        # Wrap get to count calls
        original_get = inner.get

        async def counted_get(id):
            nonlocal call_count
            call_count += 1
            return await original_get(id)

        inner.get = counted_get

        # Now create cached wrapper
        repo = CachedRepository(inner)

        # First get goes to inner repo (not in cache yet)
        result1 = await repo.get("user-1")
        assert call_count == 1

        # Second get uses cache (no additional call to inner)
        result2 = await repo.get("user-1")
        assert call_count == 1  # Still 1, used cache

        assert result1.name == "Alice"
        assert result2.name == "Alice"

    @pytest.mark.asyncio
    async def test_update_invalidates_cache(self):
        """Test update refreshes cache."""
        inner = InMemoryRepository[User]()
        repo = CachedRepository(inner)

        user = User(id="user-1", name="Alice")
        await repo.add(user)

        # Cache it
        await repo.get("user-1")

        # Update
        user.name = "Alice Updated"
        await repo.update(user)

        # Cache should have new value
        result = await repo.get("user-1")
        assert result.name == "Alice Updated"

    @pytest.mark.asyncio
    async def test_delete_removes_from_cache(self):
        """Test delete removes from cache."""
        inner = InMemoryRepository[User]()
        repo = CachedRepository(inner)

        user = User(id="user-1", name="Alice")
        await repo.add(user)
        await repo.get("user-1")  # Cache it

        await repo.delete("user-1")

        assert await repo.exists("user-1") is False

    def test_clear_cache(self):
        """Test clearing cache."""
        inner = InMemoryRepository[User]()
        repo = CachedRepository(inner)

        repo._cache["test"] = (User(), 0)
        repo.clear_cache()

        assert len(repo._cache) == 0


# =============================================================================
# ReadOnlyRepository Tests
# =============================================================================


class TestReadOnlyRepository:
    """Tests for ReadOnlyRepository."""

    @pytest.mark.asyncio
    async def test_read_operations(self):
        """Test read operations work."""
        inner = InMemoryRepository[User]()
        await inner.add(User(id="user-1", name="Alice"))
        await inner.add(User(id="user-2", name="Bob"))

        repo = ReadOnlyRepository(inner)

        # All read operations should work
        assert await repo.exists("user-1")
        assert (await repo.get("user-1")).name == "Alice"
        assert await repo.count() == 2
        assert len(await repo.list()) == 2

    @pytest.mark.asyncio
    async def test_no_write_operations(self):
        """Test no write methods available."""
        inner = InMemoryRepository[User]()
        repo = ReadOnlyRepository(inner)

        # These methods should not exist
        assert not hasattr(repo, "add")
        assert not hasattr(repo, "update")
        assert not hasattr(repo, "delete")


# =============================================================================
# Factory Tests
# =============================================================================


class TestFactory:
    """Tests for factory functions."""

    def test_create_memory_repository(self):
        """Test creating memory repository."""
        repo = create_repository(User, "memory")

        assert isinstance(repo, InMemoryRepository)

    def test_create_sqlite_repository(self):
        """Test creating SQLite repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            repo = create_repository(User, "sqlite", db_path=db_path, table_name="users")

            assert isinstance(repo, SQLiteRepository)

    def test_create_unknown_backend_raises(self):
        """Test unknown backend raises error."""
        with pytest.raises(ValueError):
            create_repository(User, "unknown")

    @pytest.mark.asyncio
    async def test_create_cached_repository(self):
        """Test creating cached repository."""
        inner = InMemoryRepository[User]()
        repo = create_cached_repository(inner, cache_ttl=60)

        assert isinstance(repo, CachedRepository)

        user = User(id="user-1", name="Alice")
        await repo.add(user)
        result = await repo.get("user-1")

        assert result.name == "Alice"
