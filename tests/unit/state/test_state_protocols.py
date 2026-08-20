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

"""Unit tests for state management protocols."""

import pytest

from victor.state.protocols import IStateManager, IStateObserver, StateScope

# =============================================================================
# Test StateScope Enum
# =============================================================================


class TestStateScope:
    """Test StateScope enumeration."""

    def test_state_scope_values(self):
        """Test StateScope has correct values."""
        assert StateScope.WORKFLOW.value == "workflow"
        assert StateScope.CONVERSATION.value == "conversation"
        assert StateScope.TEAM.value == "team"
        assert StateScope.GLOBAL.value == "global"

    def test_state_scope_comparison(self):
        """Test StateScope enum comparison."""
        assert StateScope.WORKFLOW == "workflow"
        assert StateScope.CONVERSATION == "conversation"
        assert StateScope.TEAM == "team"
        assert StateScope.GLOBAL == "global"

    def test_state_scope_enum_members(self):
        """Test StateScope enum has all expected members."""
        assert hasattr(StateScope, "WORKFLOW")
        assert hasattr(StateScope, "CONVERSATION")
        assert hasattr(StateScope, "TEAM")
        assert hasattr(StateScope, "GLOBAL")


# =============================================================================
# Test IStateManager Protocol
# =============================================================================


class MockStateManager:
    """Mock implementation of IStateManager for testing."""

    def __init__(self):
        self.scope = StateScope.WORKFLOW
        self._state = {}

    async def get(self, key: str, default=None):
        return self._state.get(key, default)

    async def set(self, key: str, value):
        self._state[key] = value

    async def delete(self, key: str):
        if key in self._state:
            del self._state[key]

    async def exists(self, key: str) -> bool:
        return key in self._state

    async def keys(self, pattern: str = "*") -> list:
        if pattern == "*":
            return list(self._state.keys())
        import fnmatch

        return [k for k in self._state.keys() if fnmatch.fnmatch(k, pattern)]

    async def get_all(self) -> dict:
        return dict(self._state)

    async def update(self, updates: dict):
        self._state.update(updates)

    async def clear(self):
        self._state.clear()

    async def snapshot(self) -> dict:
        return dict(self._state)

    async def restore(self, snapshot: dict):
        self._state = dict(snapshot)

    def add_observer(self, observer):
        pass

    def remove_observer(self, observer):
        pass


class TestIStateManagerProtocol:
    """Test IStateManager protocol compliance."""

    @pytest.mark.asyncio
    async def test_mock_manager_complies_with_protocol(self):
        """Test mock manager implements IStateManager protocol."""
        manager = MockStateManager()

        # Check protocol compliance
        assert isinstance(manager, IStateManager)

    @pytest.mark.asyncio
    async def test_manager_has_scope_attribute(self):
        """Test manager has required scope attribute."""
        manager = MockStateManager()
        assert manager.scope == StateScope.WORKFLOW

    @pytest.mark.asyncio
    async def test_get_set_delete_operations(self):
        """Test basic CRUD operations."""
        manager = MockStateManager()

        # Set value
        await manager.set("key1", "value1")
        assert await manager.exists("key1")

        # Get value
        assert await manager.get("key1") == "value1"
        assert await manager.get("key1", default="default") == "value1"
        assert await manager.get("nonexistent", default="default") == "default"

        # Delete value
        await manager.delete("key1")
        assert not await manager.exists("key1")

    @pytest.mark.asyncio
    async def test_keys_method(self):
        """Test keys() method."""
        manager = MockStateManager()

        await manager.set("key1", "value1")
        await manager.set("key2", "value2")
        await manager.set("key3", "value3")

        keys = await manager.keys()
        assert set(keys) == {"key1", "key2", "key3"}

        # Test pattern matching
        keys = await manager.keys(pattern="key*")
        assert set(keys) == {"key1", "key2", "key3"}

        keys = await manager.keys(pattern="key1")
        assert keys == ["key1"]

    @pytest.mark.asyncio
    async def test_get_all_method(self):
        """Test get_all() method."""
        manager = MockStateManager()

        await manager.set("key1", "value1")
        await manager.set("key2", "value2")

        all_state = await manager.get_all()
        assert all_state == {"key1": "value1", "key2": "value2"}

    @pytest.mark.asyncio
    async def test_update_method(self):
        """Test update() method."""
        manager = MockStateManager()

        await manager.update({"key1": "value1", "key2": "value2"})

        assert await manager.get("key1") == "value1"
        assert await manager.get("key2") == "value2"

    @pytest.mark.asyncio
    async def test_clear_method(self):
        """Test clear() method."""
        manager = MockStateManager()

        await manager.set("key1", "value1")
        await manager.set("key2", "value2")

        await manager.clear()

        assert await manager.get_all() == {}

    @pytest.mark.asyncio
    async def test_snapshot_restore_methods(self):
        """Test snapshot() and restore() methods."""
        manager = MockStateManager()

        await manager.set("key1", "value1")
        await manager.set("key2", "value2")

        # Create snapshot
        snapshot = await manager.snapshot()
        assert snapshot == {"key1": "value1", "key2": "value2"}

        # Clear and restore
        await manager.clear()
        assert await manager.get_all() == {}

        await manager.restore(snapshot)
        assert await manager.get("key1") == "value1"
        assert await manager.get("key2") == "value2"

    @pytest.mark.asyncio
    async def test_add_remove_observers(self):
        """Test add_observer() and remove_observer() methods."""
        manager = MockStateManager()

        # Mock observer
        class MockObserver:
            async def on_state_changed(self, scope, key, old_value, new_value, metadata=None):
                pass

        observer = MockObserver()

        # These should not raise errors
        manager.add_observer(observer)
        manager.remove_observer(observer)


# =============================================================================
# Test IStateObserver Protocol
# =============================================================================


class MockStateObserver:
    """Mock implementation of IStateObserver for testing."""

    def __init__(self):
        self.notifications = []

    async def on_state_changed(self, scope, key, old_value, new_value, metadata=None):
        self.notifications.append(
            {
                "scope": scope,
                "key": key,
                "old_value": old_value,
                "new_value": new_value,
                "metadata": metadata,
            }
        )


class TestIStateObserverProtocol:
    """Test IStateObserver protocol compliance."""

    @pytest.mark.asyncio
    async def test_mock_observer_complies_with_protocol(self):
        """Test mock observer implements IStateObserver protocol."""
        observer = MockStateObserver()

        # Check protocol compliance
        assert isinstance(observer, IStateObserver)

    @pytest.mark.asyncio
    async def test_on_state_changed_method(self):
        """Test on_state_changed() method."""
        observer = MockStateObserver()

        await observer.on_state_changed(
            scope=StateScope.WORKFLOW,
            key="test_key",
            old_value=None,
            new_value="new_value",
            metadata={"source": "test"},
        )

        assert len(observer.notifications) == 1
        notification = observer.notifications[0]

        assert notification["scope"] == StateScope.WORKFLOW
        assert notification["key"] == "test_key"
        assert notification["old_value"] is None
        assert notification["new_value"] == "new_value"
        assert notification["metadata"] == {"source": "test"}


# =============================================================================
# Integration Tests
# =============================================================================


class TestProtocolIntegration:
    """Integration tests for protocols."""

    @pytest.mark.asyncio
    async def test_manager_observer_integration(self):
        """Test manager and observer integration."""
        manager = MockStateManager()
        observer = MockStateObserver()

        # Add observer to manager
        manager.add_observer(observer)

        # Set value (would normally trigger observer)
        await manager.set("key1", "value1")

        # In a real implementation, this would trigger observer.on_state_changed()
        # For now, we just verify the observer has the required method
        assert hasattr(observer, "on_state_changed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
