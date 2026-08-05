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

"""Unit tests for state manager implementations."""

import pytest

from victor.state.managers import (
    ConversationStateManager,
    GlobalStateManagerImpl,
    TeamStateManager,
    WorkflowStateManager,
)
from victor.state.protocols import IStateManager, IStateObserver, StateScope

# =============================================================================
# Mock Observer for Testing
# =============================================================================


class MockStateObserver:
    """Mock observer for testing state change notifications."""

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


# =============================================================================
# Test WorkflowStateManager
# =============================================================================


class TestWorkflowStateManager:
    """Test WorkflowStateManager class."""

    def test_create_manager(self):
        """Test creating a workflow state manager."""
        manager = WorkflowStateManager()

        assert manager.scope == StateScope.WORKFLOW
        assert manager._state == {}
        assert manager._observers == []

    @pytest.mark.asyncio
    async def test_manager_complies_with_protocol(self):
        """Test manager implements IStateManager protocol."""
        manager = WorkflowStateManager()

        # Check protocol compliance
        assert isinstance(manager, IStateManager)

    @pytest.mark.asyncio
    async def test_get_set_operations(self):
        """Test basic get/set operations."""
        manager = WorkflowStateManager()

        # Set value
        await manager.set("task_id", "task-123")
        assert await manager.exists("task_id")

        # Get value
        assert await manager.get("task_id") == "task-123"
        assert await manager.get("task_id", default="default") == "task-123"
        assert await manager.get("nonexistent", default="default") == "default"

    @pytest.mark.asyncio
    async def test_delete_operations(self):
        """Test delete operations."""
        manager = WorkflowStateManager()

        await manager.set("key1", "value1")
        assert await manager.exists("key1")

        await manager.delete("key1")
        assert not await manager.exists("key1")

        # Delete non-existent key should not raise
        await manager.delete("nonexistent")

    @pytest.mark.asyncio
    async def test_keys_method(self):
        """Test keys() method."""
        manager = WorkflowStateManager()

        await manager.set("task_id", "task-123")
        await manager.set("workflow_status", "running")
        await manager.set("agent_id", "agent-1")

        keys = await manager.keys()
        assert set(keys) == {"task_id", "workflow_status", "agent_id"}

        # Test pattern matching
        keys = await manager.keys(pattern="task_*")
        assert set(keys) == {"task_id"}

        keys = await manager.keys(pattern="*status")
        assert keys == ["workflow_status"]

    @pytest.mark.asyncio
    async def test_get_all_method(self):
        """Test get_all() method."""
        manager = WorkflowStateManager()

        await manager.set("key1", "value1")
        await manager.set("key2", "value2")

        all_state = await manager.get_all()
        assert all_state == {"key1": "value1", "key2": "value2"}

    @pytest.mark.asyncio
    async def test_update_method(self):
        """Test update() method."""
        manager = WorkflowStateManager()

        await manager.update({"key1": "value1", "key2": "value2"})

        assert await manager.get("key1") == "value1"
        assert await manager.get("key2") == "value2"

    @pytest.mark.asyncio
    async def test_clear_method(self):
        """Test clear() method."""
        manager = WorkflowStateManager()

        await manager.set("key1", "value1")
        await manager.set("key2", "value2")

        await manager.clear()

        assert await manager.get_all() == {}

    @pytest.mark.asyncio
    async def test_snapshot_restore_methods(self):
        """Test snapshot() and restore() methods."""
        manager = WorkflowStateManager()

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
    async def test_observer_notifications(self):
        """Test observer receives notifications."""
        manager = WorkflowStateManager()
        observer = MockStateObserver()

        manager.add_observer(observer)

        # Set value
        await manager.set("key1", "value1")

        assert len(observer.notifications) == 1
        notification = observer.notifications[0]
        assert notification["scope"] == StateScope.WORKFLOW
        assert notification["key"] == "key1"
        assert notification["old_value"] is None
        assert notification["new_value"] == "value1"

        # Update value
        await manager.set("key1", "value1-updated")

        assert len(observer.notifications) == 2
        notification = observer.notifications[1]
        assert notification["old_value"] == "value1"
        assert notification["new_value"] == "value1-updated"

        # Delete value
        await manager.delete("key1")

        assert len(observer.notifications) == 3
        notification = observer.notifications[2]
        assert notification["old_value"] == "value1-updated"
        assert notification["new_value"] is None

    @pytest.mark.asyncio
    async def test_add_remove_observers(self):
        """Test adding and removing observers."""
        manager = WorkflowStateManager()
        observer = MockStateObserver()

        manager.add_observer(observer)
        assert observer in manager._observers

        manager.remove_observer(observer)
        assert observer not in manager._observers

        # Remove again should not raise
        manager.remove_observer(observer)

    @pytest.mark.asyncio
    async def test_multiple_observers(self):
        """Test multiple observers receive notifications."""
        manager = WorkflowStateManager()
        observer1 = MockStateObserver()
        observer2 = MockStateObserver()

        manager.add_observer(observer1)
        manager.add_observer(observer2)

        await manager.set("key1", "value1")

        assert len(observer1.notifications) == 1
        assert len(observer2.notifications) == 1


# =============================================================================
# Test ConversationStateManager
# =============================================================================


class TestConversationStateManager:
    """Test ConversationStateManager class."""

    def test_create_manager(self):
        """Test creating a conversation state manager."""
        manager = ConversationStateManager()

        assert manager.scope == StateScope.CONVERSATION
        assert manager._state == {}
        assert manager._observers == []

    @pytest.mark.asyncio
    async def test_manager_complies_with_protocol(self):
        """Test manager implements IStateManager protocol."""
        manager = ConversationStateManager()

        # Check protocol compliance
        assert isinstance(manager, IStateManager)

    @pytest.mark.asyncio
    async def test_conversation_stage_management(self):
        """Test conversation stage tracking."""
        manager = ConversationStateManager()

        # Set initial stage
        await manager.set("stage", "gathering")
        assert await manager.get("stage") == "gathering"

        # Update stage
        await manager.set("stage", "processing")
        assert await manager.get("stage") == "processing"

        # Add conversation history
        await manager.set("history", [])
        history = await manager.get("history")
        assert history == []

    @pytest.mark.asyncio
    async def test_keys_method(self):
        """Test keys() method."""
        manager = ConversationStateManager()

        await manager.set("stage", "gathering")
        await manager.set("turn_count", 5)
        await manager.set("user_input", "Hello")

        keys = await manager.keys()
        assert set(keys) == {"stage", "turn_count", "user_input"}

        # Test pattern matching
        keys = await manager.keys(pattern="*count")
        assert keys == ["turn_count"]


# =============================================================================
# Test TeamStateManager
# =============================================================================


class TestTeamStateManager:
    """Test TeamStateManager class."""

    def test_create_manager(self):
        """Test creating a team state manager."""
        manager = TeamStateManager()

        assert manager.scope == StateScope.TEAM
        assert manager._state == {}
        assert manager._observers == []

    @pytest.mark.asyncio
    async def test_manager_complies_with_protocol(self):
        """Test manager implements IStateManager protocol."""
        manager = TeamStateManager()

        # Check protocol compliance
        assert isinstance(manager, IStateManager)

    @pytest.mark.asyncio
    async def test_coordinator_tracking(self):
        """Test coordinator agent tracking."""
        manager = TeamStateManager()

        await manager.set("coordinator", "agent-1")
        assert await manager.get("coordinator") == "agent-1"

        # Switch coordinator
        await manager.set("coordinator", "agent-2")
        assert await manager.get("coordinator") == "agent-2"

    @pytest.mark.asyncio
    async def test_team_members_tracking(self):
        """Test team members tracking."""
        manager = TeamStateManager()

        await manager.set("members", ["agent-1", "agent-2", "agent-3"])
        members = await manager.get("members")

        assert members == ["agent-1", "agent-2", "agent-3"]

    @pytest.mark.asyncio
    async def test_shared_state(self):
        """Test shared state between team members."""
        manager = TeamStateManager()

        await manager.set("shared_context", {"task": "analyze data"})
        context = await manager.get("shared_context")

        assert context == {"task": "analyze data"}


# =============================================================================
# Test GlobalStateManagerImpl
# =============================================================================


class TestGlobalStateManagerImpl:
    """Test GlobalStateManagerImpl class."""

    def test_create_manager(self):
        """Test creating a global state manager."""
        manager = GlobalStateManagerImpl()

        assert manager.scope == StateScope.GLOBAL
        assert manager._state == {}
        assert manager._observers == []

    @pytest.mark.asyncio
    async def test_manager_complies_with_protocol(self):
        """Test manager implements IStateManager protocol."""
        manager = GlobalStateManagerImpl()

        # Check protocol compliance
        assert isinstance(manager, IStateManager)

    @pytest.mark.asyncio
    async def test_configuration_storage(self):
        """Test configuration storage."""
        manager = GlobalStateManagerImpl()

        config = {"debug": True, "log_level": "INFO"}
        await manager.set("config", config)

        retrieved_config = await manager.get("config")
        assert retrieved_config == config

    @pytest.mark.asyncio
    async def test_global_settings(self):
        """Test global settings management."""
        manager = GlobalStateManagerImpl()

        await manager.set("max_tokens", 4096)
        await manager.set("temperature", 0.7)

        assert await manager.get("max_tokens") == 4096
        assert await manager.get("temperature") == 0.7

    @pytest.mark.asyncio
    async def test_snapshot_restore_across_scopes(self):
        """Test snapshot and restore preserves state."""
        manager = GlobalStateManagerImpl()

        # Set up global state
        await manager.set("app_name", "Victor")
        await manager.set("version", "1.0.0")
        await manager.set("debug", True)

        # Create snapshot
        snapshot = await manager.snapshot()
        assert snapshot == {
            "app_name": "Victor",
            "version": "1.0.0",
            "debug": True,
        }

        # Modify state
        await manager.set("version", "2.0.0")

        # Restore from snapshot
        await manager.restore(snapshot)

        assert await manager.get("version") == "1.0.0"
        assert await manager.get("app_name") == "Victor"
        assert await manager.get("debug") is True


# =============================================================================
# Integration Tests
# =============================================================================


class TestStateManagerIntegration:
    """Integration tests for state managers."""

    @pytest.mark.asyncio
    async def test_multiple_managers_independent_state(self):
        """Test multiple managers maintain independent state."""
        workflow_mgr = WorkflowStateManager()
        conversation_mgr = ConversationStateManager()
        team_mgr = TeamStateManager()

        # Set same key in different managers
        await workflow_mgr.set("status", "running")
        await conversation_mgr.set("status", "gathering")
        await team_mgr.set("status", "coordinating")

        # Verify each manager has its own value
        assert await workflow_mgr.get("status") == "running"
        assert await conversation_mgr.get("status") == "gathering"
        assert await team_mgr.get("status") == "coordinating"

    @pytest.mark.asyncio
    async def test_observer_notified_on_all_operations(self):
        """Test observer is notified for set, delete, and update."""
        manager = WorkflowStateManager()
        observer = MockStateObserver()

        manager.add_observer(observer)

        # Set operation
        await manager.set("key1", "value1")
        assert len(observer.notifications) == 1

        # Update operation (another set)
        await manager.set("key1", "value2")
        assert len(observer.notifications) == 2

        # Delete operation
        await manager.delete("key1")
        assert len(observer.notifications) == 3

        # Verify notification types
        assert observer.notifications[0]["new_value"] == "value1"
        assert observer.notifications[1]["old_value"] == "value1"
        assert observer.notifications[1]["new_value"] == "value2"
        assert observer.notifications[2]["old_value"] == "value2"
        assert observer.notifications[2]["new_value"] is None

    @pytest.mark.asyncio
    async def test_update_triggers_multiple_notifications(self):
        """Test update() method triggers notification per key."""
        manager = ConversationStateManager()
        observer = MockStateObserver()

        manager.add_observer(observer)

        # Update multiple keys
        await manager.update({"key1": "value1", "key2": "value2", "key3": "value3"})

        # Should trigger 3 notifications (one per key)
        assert len(observer.notifications) == 3

        keys = [n["key"] for n in observer.notifications]
        assert keys == ["key1", "key2", "key3"]

    @pytest.mark.asyncio
    async def test_snapshot_restore_independence(self):
        """Test snapshots are independent and can be restored."""
        manager = TeamStateManager()

        # Create initial snapshot
        await manager.set("coordinator", "agent-1")
        snapshot1 = await manager.snapshot()

        # Modify state
        await manager.set("coordinator", "agent-2")
        snapshot2 = await manager.snapshot()

        # Restore to first snapshot
        await manager.restore(snapshot1)
        assert await manager.get("coordinator") == "agent-1"

        # Restore to second snapshot
        await manager.restore(snapshot2)
        assert await manager.get("coordinator") == "agent-2"

    @pytest.mark.asyncio
    async def test_clear_clears_all_state(self):
        """Test clear() removes all state."""
        manager = GlobalStateManagerImpl()

        # Add multiple keys
        await manager.update({"key1": "value1", "key2": "value2", "key3": "value3"})

        # Clear all
        await manager.clear()

        # Verify all gone
        assert await manager.get_all() == {}
        assert not await manager.exists("key1")
        assert not await manager.exists("key2")
        assert not await manager.exists("key3")

    @pytest.mark.asyncio
    async def test_pattern_matching_in_keys(self):
        """Test various pattern matching scenarios."""
        manager = WorkflowStateManager()

        await manager.update(
            {
                "task_id": "123",
                "task_name": "test",
                "workflow_status": "running",
                "workflow_step": "5",
                "agent_id": "agent-1",
            }
        )

        # Match all
        all_keys = await manager.keys("*")
        assert len(all_keys) == 5

        # Match task_*
        task_keys = await manager.keys("task_*")
        assert set(task_keys) == {"task_id", "task_name"}

        # Match workflow_*
        workflow_keys = await manager.keys("workflow_*")
        assert set(workflow_keys) == {"workflow_status", "workflow_step"}

        # Match specific pattern
        agent_keys = await manager.keys("agent_*")
        assert agent_keys == ["agent_id"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
