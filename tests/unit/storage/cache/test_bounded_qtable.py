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

"""Unit tests for BoundedQTable."""

import pytest

from victor.storage.cache.rl_eviction_policy import BoundedQTable


class TestBoundedQTableInitialization:
    """Tests for BoundedQTable initialization."""

    def test_init_default_max_size(self):
        """Test initialization with default max_size."""
        table = BoundedQTable()
        assert table.max_size == 100000
        assert len(table) == 0

    def test_init_custom_max_size(self):
        """Test initialization with custom max_size."""
        table = BoundedQTable(max_size=50)
        assert table.max_size == 50
        assert len(table) == 0

    def test_init_empty_table(self):
        """Test that table starts empty."""
        table = BoundedQTable()
        assert len(table) == 0
        assert table.keys() == []


class TestBoundedQTableGet:
    """Tests for BoundedQTable.get method."""

    def test_get_existing_key(self):
        """Test getting an existing key returns the value."""
        table = BoundedQTable()
        table.set("key1", 0.5)
        assert table.get("key1") == 0.5

    def test_get_non_existing_key_returns_default(self):
        """Test getting a non-existing key returns default value."""
        table = BoundedQTable()
        assert table.get("nonexistent") == 0.0

    def test_get_non_existing_key_custom_default(self):
        """Test getting a non-existing key with custom default."""
        table = BoundedQTable()
        assert table.get("nonexistent", default=1.0) == 1.0

    def test_get_updates_access_order(self):
        """Test that get updates access order for LRU tracking."""
        table = BoundedQTable(max_size=3)
        table.set("key1", 0.1)
        table.set("key2", 0.2)
        table.set("key3", 0.3)

        # Access key1, moving it to the end of access order
        table.get("key1")

        # Add a new key, which should evict key2 (now least recently used)
        table.set("key4", 0.4)

        assert "key1" in table  # Should still exist (was accessed)
        assert "key2" not in table  # Should be evicted (LRU)
        assert "key3" in table
        assert "key4" in table


class TestBoundedQTableSet:
    """Tests for BoundedQTable.set method."""

    def test_set_new_item(self):
        """Test setting a new item adds it to the table."""
        table = BoundedQTable()
        table.set("key1", 0.5)
        assert "key1" in table
        assert table.get("key1") == 0.5
        assert len(table) == 1

    def test_set_update_existing_item(self):
        """Test updating an existing item."""
        table = BoundedQTable()
        table.set("key1", 0.5)
        table.set("key1", 0.9)
        assert table.get("key1") == 0.9
        assert len(table) == 1

    def test_set_multiple_items(self):
        """Test setting multiple items."""
        table = BoundedQTable()
        table.set("key1", 0.1)
        table.set("key2", 0.2)
        table.set("key3", 0.3)
        assert len(table) == 3
        assert table.get("key1") == 0.1
        assert table.get("key2") == 0.2
        assert table.get("key3") == 0.3


class TestBoundedQTableLRUEviction:
    """Tests for LRU eviction behavior."""

    def test_lru_eviction_when_max_size_exceeded(self):
        """Test that LRU entry is evicted when max_size is exceeded."""
        table = BoundedQTable(max_size=3)
        table.set("key1", 0.1)
        table.set("key2", 0.2)
        table.set("key3", 0.3)

        # Adding a 4th item should evict key1 (least recently used)
        table.set("key4", 0.4)

        assert len(table) == 3
        assert "key1" not in table  # Evicted
        assert "key2" in table
        assert "key3" in table
        assert "key4" in table

    def test_lru_eviction_respects_access_order(self):
        """Test that eviction respects access order."""
        table = BoundedQTable(max_size=3)
        table.set("key1", 0.1)
        table.set("key2", 0.2)
        table.set("key3", 0.3)

        # Access key1, making key2 the LRU
        _ = table.get("key1")

        # Adding a 4th item should evict key2 (now LRU)
        table.set("key4", 0.4)

        assert "key1" in table  # Was accessed, not evicted
        assert "key2" not in table  # LRU, evicted
        assert "key3" in table
        assert "key4" in table

    def test_lru_eviction_with_update(self):
        """Test that updating an item updates its access order."""
        table = BoundedQTable(max_size=3)
        table.set("key1", 0.1)
        table.set("key2", 0.2)
        table.set("key3", 0.3)

        # Update key1, making key2 the LRU
        table.set("key1", 0.15)

        # Adding a 4th item should evict key2 (now LRU)
        table.set("key4", 0.4)

        assert "key1" in table  # Was updated, not evicted
        assert table.get("key1") == 0.15
        assert "key2" not in table  # LRU, evicted
        assert "key3" in table
        assert "key4" in table

    def test_eviction_maintains_correct_size(self):
        """Test that eviction maintains correct table size."""
        table = BoundedQTable(max_size=5)

        # Add more items than max_size
        for i in range(10):
            table.set(f"key{i}", float(i))

        assert len(table) == 5
        # Only the last 5 should remain
        for i in range(5):
            assert f"key{i}" not in table
        for i in range(5, 10):
            assert f"key{i}" in table


class TestBoundedQTableDunderMethods:
    """Tests for dunder methods (__len__, __contains__)."""

    def test_len_empty_table(self):
        """Test __len__ on empty table."""
        table = BoundedQTable()
        assert len(table) == 0

    def test_len_with_items(self):
        """Test __len__ with items."""
        table = BoundedQTable()
        table.set("key1", 0.1)
        table.set("key2", 0.2)
        assert len(table) == 2

    def test_contains_existing_key(self):
        """Test __contains__ for existing key."""
        table = BoundedQTable()
        table.set("key1", 0.5)
        assert "key1" in table

    def test_contains_non_existing_key(self):
        """Test __contains__ for non-existing key."""
        table = BoundedQTable()
        assert "nonexistent" not in table


class TestBoundedQTableClear:
    """Tests for BoundedQTable.clear method."""

    def test_clear_empties_table(self):
        """Test that clear empties the table."""
        table = BoundedQTable()
        table.set("key1", 0.1)
        table.set("key2", 0.2)
        table.set("key3", 0.3)

        table.clear()

        assert len(table) == 0
        assert "key1" not in table
        assert "key2" not in table
        assert "key3" not in table

    def test_clear_empties_access_order(self):
        """Test that clear also resets access order."""
        table = BoundedQTable(max_size=3)
        table.set("key1", 0.1)
        table.set("key2", 0.2)

        table.clear()

        # After clear, adding 3 items should not cause eviction
        table.set("a", 1.0)
        table.set("b", 2.0)
        table.set("c", 3.0)

        assert len(table) == 3
        assert "a" in table
        assert "b" in table
        assert "c" in table


class TestBoundedQTableKeys:
    """Tests for BoundedQTable.keys method."""

    def test_keys_empty_table(self):
        """Test keys() on empty table."""
        table = BoundedQTable()
        assert table.keys() == []

    def test_keys_with_items(self):
        """Test keys() returns all keys."""
        table = BoundedQTable()
        table.set("key1", 0.1)
        table.set("key2", 0.2)
        table.set("key3", 0.3)

        keys = table.keys()
        assert len(keys) == 3
        assert "key1" in keys
        assert "key2" in keys
        assert "key3" in keys

    def test_keys_returns_list_copy(self):
        """Test that keys() returns a copy, not the internal structure."""
        table = BoundedQTable()
        table.set("key1", 0.1)

        keys = table.keys()
        keys.append("key2")

        # Original table should not be modified
        assert "key2" not in table
        assert len(table) == 1
