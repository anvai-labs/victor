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

"""Unit tests for GlobalStateManager and factory functions."""

import logging

import pytest

from victor.state.factory import (
    get_global_manager,
    get_tracer,
    initialize_with_existing,
    is_initialized,
    reset_global_manager,
    set_tracer,
)
from victor.state.global_state_manager import GlobalStateManager
from victor.state.managers import (
    ConversationStateManager,
    GlobalStateManagerImpl,
    TeamStateManager,
    WorkflowStateManager,
)
from victor.state.protocols import StateScope
from victor.state.tracer import StateTracer

# =============================================================================
# Test GlobalStateManager
# =============================================================================


class TestGlobalStateManager:
    """Test GlobalStateManager class."""

    def test_create_manager(self):
        """Test creating a global state manager."""
        manager = GlobalStateManager()

        assert manager._managers == {}
        assert manager._tracer is None

    def test_register_manager(self):
        """Test registering a state manager."""
        global_manager = GlobalStateManager()
        workflow_manager = WorkflowStateManager()

        global_manager.register_manager(StateScope.WORKFLOW, workflow_manager)

        assert StateScope.WORKFLOW in global_manager._managers
        assert global_manager._managers[StateScope.WORKFLOW] == workflow_manager

    def test_register_manager_replaces_existing(self):
        """Test registering a manager replaces existing one."""
        global_manager = GlobalStateManager()
        manager1 = WorkflowStateManager()
        manager2 = WorkflowStateManager()

        global_manager.register_manager(StateScope.WORKFLOW, manager1)
        global_manager.register_manager(StateScope.WORKFLOW, manager2)

        # Should replace with new manager
        assert global_manager._managers[StateScope.WORKFLOW] == manager2

    def test_set_tracer(self):
        """Test setting the state tracer."""
        from victor.core.events import ObservabilityBus, InMemoryEventBackend

        global_manager = GlobalStateManager()
        backend = InMemoryEventBackend()
        event_bus = ObservabilityBus(backend=backend)
        tracer = StateTracer(event_bus)

        global_manager.set_tracer(tracer)

        assert global_manager._tracer == tracer

    @pytest.mark.asyncio
    async def test_get_set_operations(self):
        """Test basic get/set operations across scopes."""
        global_manager = GlobalStateManager()
        global_manager.register_manager(StateScope.WORKFLOW, WorkflowStateManager())
        global_manager.register_manager(StateScope.CONVERSATION, ConversationStateManager())

        # Set values in different scopes
        await global_manager.set("task_id", "task-123", scope=StateScope.WORKFLOW)
        await global_manager.set("stage", "gathering", scope=StateScope.CONVERSATION)

        # Get values from different scopes
        assert await global_manager.get("task_id", scope=StateScope.WORKFLOW) == "task-123"
        assert await global_manager.get("stage", scope=StateScope.CONVERSATION) == "gathering"

    @pytest.mark.asyncio
    async def test_get_with_default_scope(self):
        """Test get operation defaults to GLOBAL scope."""
        global_manager = GlobalStateManager()
        global_manager.register_manager(StateScope.GLOBAL, GlobalStateManagerImpl())

        await global_manager.set("config", {"debug": True})

        # Should default to GLOBAL scope
        assert await global_manager.get("config") == {"debug": True}

    @pytest.mark.asyncio
    async def test_get_fails_on_unregistered_scope(self):
        """Test get raises ValueError for unregistered scope."""
        global_manager = GlobalStateManager()

        with pytest.raises(ValueError, match="No manager registered"):
            await global_manager.get("key", scope=StateScope.WORKFLOW)

    @pytest.mark.asyncio
    async def test_delete_operations(self):
        """Test delete operations across scopes."""
        global_manager = GlobalStateManager()
        global_manager.register_manager(StateScope.WORKFLOW, WorkflowStateManager())

        await global_manager.set("key1", "value1", scope=StateScope.WORKFLOW)
        assert await global_manager.exists("key1", scope=StateScope.WORKFLOW)

        await global_manager.delete("key1", scope=StateScope.WORKFLOW)
        assert not await global_manager.exists("key1", scope=StateScope.WORKFLOW)

    @pytest.mark.asyncio
    async def test_keys_method(self):
        """Test keys() method across scopes."""
        global_manager = GlobalStateManager()
        global_manager.register_manager(StateScope.WORKFLOW, WorkflowStateManager())
        global_manager.register_manager(StateScope.TEAM, TeamStateManager())

        await global_manager.set("task_id", "task-123", scope=StateScope.WORKFLOW)
        await global_manager.set("status", "running", scope=StateScope.WORKFLOW)
        await global_manager.set("coordinator", "agent-1", scope=StateScope.TEAM)

        # Get keys from workflow scope
        workflow_keys = await global_manager.keys(scope=StateScope.WORKFLOW, pattern="*")
        assert set(workflow_keys) == {"task_id", "status"}

        # Get keys from team scope
        team_keys = await global_manager.keys(scope=StateScope.TEAM, pattern="*")
        assert team_keys == ["coordinator"]

    @pytest.mark.asyncio
    async def test_get_all_method(self):
        """Test get_all() method across scopes."""
        global_manager = GlobalStateManager()
        global_manager.register_manager(StateScope.CONVERSATION, ConversationStateManager())

        await global_manager.set("stage", "gathering", scope=StateScope.CONVERSATION)
        await global_manager.set("turn_count", 5, scope=StateScope.CONVERSATION)

        all_state = await global_manager.get_all(scope=StateScope.CONVERSATION)
        assert all_state == {"stage": "gathering", "turn_count": 5}

    @pytest.mark.asyncio
    async def test_update_method(self):
        """Test update() method across scopes."""
        global_manager = GlobalStateManager()
        global_manager.register_manager(StateScope.GLOBAL, GlobalStateManagerImpl())

        await global_manager.update({"debug": True, "log_level": "INFO"}, scope=StateScope.GLOBAL)

        assert await global_manager.get("debug", scope=StateScope.GLOBAL) is True
        assert await global_manager.get("log_level", scope=StateScope.GLOBAL) == "INFO"

    @pytest.mark.asyncio
    async def test_clear_method(self):
        """Test clear() method across scopes."""
        global_manager = GlobalStateManager()
        global_manager.register_manager(StateScope.TEAM, TeamStateManager())

        await global_manager.set("coordinator", "agent-1", scope=StateScope.TEAM)
        await global_manager.set("members", ["agent-1", "agent-2"], scope=StateScope.TEAM)

        await global_manager.clear(scope=StateScope.TEAM)

        assert await global_manager.get_all(scope=StateScope.TEAM) == {}

    @pytest.mark.asyncio
    async def test_create_checkpoint(self):
        """Test creating checkpoint across all scopes."""
        global_manager = GlobalStateManager()
        global_manager.register_manager(StateScope.WORKFLOW, WorkflowStateManager())
        global_manager.register_manager(StateScope.CONVERSATION, ConversationStateManager())
        global_manager.register_manager(StateScope.TEAM, TeamStateManager())
        global_manager.register_manager(StateScope.GLOBAL, GlobalStateManagerImpl())

        # Set values in all scopes
        await global_manager.set("task_id", "task-123", scope=StateScope.WORKFLOW)
        await global_manager.set("stage", "gathering", scope=StateScope.CONVERSATION)
        await global_manager.set("coordinator", "agent-1", scope=StateScope.TEAM)
        await global_manager.set("debug", True, scope=StateScope.GLOBAL)

        # Create checkpoint
        checkpoint = await global_manager.create_checkpoint()

        # Verify checkpoint contains all scopes
        assert "workflow" in checkpoint
        assert "conversation" in checkpoint
        assert "team" in checkpoint
        assert "global" in checkpoint

        assert checkpoint["workflow"] == {"task_id": "task-123"}
        assert checkpoint["conversation"] == {"stage": "gathering"}
        assert checkpoint["team"] == {"coordinator": "agent-1"}
        assert checkpoint["global"] == {"debug": True}

    @pytest.mark.asyncio
    async def test_restore_checkpoint(self):
        """Test restoring checkpoint across all scopes."""
        global_manager = GlobalStateManager()
        global_manager.register_manager(StateScope.WORKFLOW, WorkflowStateManager())
        global_manager.register_manager(StateScope.CONVERSATION, ConversationStateManager())

        # Create checkpoint
        checkpoint = {
            "workflow": {"task_id": "task-123", "status": "running"},
            "conversation": {"stage": "processing", "turn_count": 3},
        }

        # Restore checkpoint
        await global_manager.restore_checkpoint(checkpoint)

        # Verify restored state
        assert await global_manager.get("task_id", scope=StateScope.WORKFLOW) == "task-123"
        assert await global_manager.get("status", scope=StateScope.WORKFLOW) == "running"
        assert await global_manager.get("stage", scope=StateScope.CONVERSATION) == "processing"
        assert await global_manager.get("turn_count", scope=StateScope.CONVERSATION) == 3

    @pytest.mark.asyncio
    async def test_restore_checkpoint_fails_on_unregistered_scope(self):
        """Test restore fails for unregistered scope."""
        global_manager = GlobalStateManager()
        global_manager.register_manager(StateScope.WORKFLOW, WorkflowStateManager())

        checkpoint = {
            "workflow": {"task_id": "task-123"},
            "conversation": {"stage": "gathering"},
        }

        with pytest.raises(ValueError, match="no manager registered"):
            await global_manager.restore_checkpoint(checkpoint)

    @pytest.mark.asyncio
    async def test_get_cross_scope_state(self):
        """Test getting state from all scopes."""
        global_manager = GlobalStateManager()
        global_manager.register_manager(StateScope.WORKFLOW, WorkflowStateManager())
        global_manager.register_manager(StateScope.CONVERSATION, ConversationStateManager())
        global_manager.register_manager(StateScope.TEAM, TeamStateManager())

        # Set values in all scopes
        await global_manager.set("task_id", "task-123", scope=StateScope.WORKFLOW)
        await global_manager.set("stage", "gathering", scope=StateScope.CONVERSATION)
        await global_manager.set("coordinator", "agent-1", scope=StateScope.TEAM)

        # Get cross-scope state
        cross_scope_state = await global_manager.get_cross_scope_state()

        assert cross_scope_state["workflow"] == {"task_id": "task-123"}
        assert cross_scope_state["conversation"] == {"stage": "gathering"}
        assert cross_scope_state["team"] == {"coordinator": "agent-1"}

    def test_get_registered_scopes(self):
        """Test getting list of registered scopes."""
        global_manager = GlobalStateManager()
        global_manager.register_manager(StateScope.WORKFLOW, WorkflowStateManager())
        global_manager.register_manager(StateScope.CONVERSATION, ConversationStateManager())

        scopes = global_manager.get_registered_scopes()

        assert StateScope.WORKFLOW in scopes
        assert StateScope.CONVERSATION in scopes
        assert StateScope.TEAM not in scopes

    def test_has_scope(self):
        """Test checking if scope is registered."""
        global_manager = GlobalStateManager()
        global_manager.register_manager(StateScope.WORKFLOW, WorkflowStateManager())

        assert global_manager.has_scope(StateScope.WORKFLOW) is True
        assert global_manager.has_scope(StateScope.CONVERSATION) is False

    def test_get_manager(self):
        """Test getting manager for specific scope."""
        global_manager = GlobalStateManager()
        workflow_manager = WorkflowStateManager()

        global_manager.register_manager(StateScope.WORKFLOW, workflow_manager)

        assert global_manager.get_manager(StateScope.WORKFLOW) == workflow_manager
        assert global_manager.get_manager(StateScope.CONVERSATION) is None

    @pytest.mark.asyncio
    async def test_operations_with_tracer(self):
        """Test operations trigger tracer when set."""
        from victor.core.events import ObservabilityBus, InMemoryEventBackend

        global_manager = GlobalStateManager()
        global_manager.register_manager(StateScope.WORKFLOW, WorkflowStateManager())

        backend = InMemoryEventBackend()
        event_bus = ObservabilityBus(backend=backend)
        tracer = StateTracer(event_bus)
        global_manager.set_tracer(tracer)

        # Set value
        await global_manager.set("key1", "value1", scope=StateScope.WORKFLOW)

        # Verify tracer recorded transition
        history = tracer.get_history()
        assert len(history) == 1
        assert history[0].scope == "workflow"
        assert history[0].key == "key1"
        assert history[0].new_value == "value1"

        # Update value
        await global_manager.set("key1", "value2", scope=StateScope.WORKFLOW)

        # Verify tracer recorded update
        history = tracer.get_history()
        assert len(history) == 2
        assert history[1].old_value == "value1"
        assert history[1].new_value == "value2"

        # Delete value
        await global_manager.delete("key1", scope=StateScope.WORKFLOW)

        # Verify tracer recorded deletion
        history = tracer.get_history()
        assert len(history) == 3
        assert history[2].old_value == "value2"
        assert history[2].new_value is None


# =============================================================================
# Test Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Test factory functions."""

    def setup_method(self):
        """Reset global manager before each test."""
        reset_global_manager()

    def teardown_method(self):
        """Reset global manager after each test."""
        reset_global_manager()

    def test_get_global_manager_creates_instance(self):
        """Test get_global_manager creates singleton instance."""
        manager1 = get_global_manager()
        manager2 = get_global_manager()

        # Should return same instance
        assert manager1 is manager2

    def test_get_global_manager_registers_all_scopes(self):
        """Test get_global_manager registers all scope managers."""
        manager = get_global_manager()

        # Check all scopes are registered
        assert manager.has_scope(StateScope.WORKFLOW)
        assert manager.has_scope(StateScope.CONVERSATION)
        assert manager.has_scope(StateScope.TEAM)
        assert manager.has_scope(StateScope.GLOBAL)

    def test_get_global_manager_returns_correct_managers(self):
        """Test get_global_manager returns correct manager types."""
        manager = get_global_manager()

        # Check manager types
        assert isinstance(manager.get_manager(StateScope.WORKFLOW), WorkflowStateManager)
        assert isinstance(
            manager.get_manager(StateScope.CONVERSATION),
            ConversationStateManager,
        )
        assert isinstance(manager.get_manager(StateScope.TEAM), TeamStateManager)
        assert isinstance(manager.get_manager(StateScope.GLOBAL), GlobalStateManagerImpl)

    def test_is_initialized(self):
        """Test is_initialized checks if manager was created."""
        # Before initialization
        assert is_initialized() is False

        # After initialization
        get_global_manager()
        assert is_initialized() is True

    def test_reset_global_manager(self):
        """Test reset_global_manager clears singleton."""
        # Create manager
        manager1 = get_global_manager()
        assert is_initialized() is True

        # Reset
        reset_global_manager()
        assert is_initialized() is False

        # Create new manager
        manager2 = get_global_manager()
        assert manager1 is not manager2

    def test_reset_global_manager_logs_debug_only(self, caplog):
        """Reset noise should stay out of normal INFO logs."""
        get_global_manager()
        caplog.set_level(logging.DEBUG, logger="victor.state.factory")

        reset_global_manager()

        assert any(
            record.levelno == logging.DEBUG and "GlobalStateManager reset" in record.message
            for record in caplog.records
        )
        assert not any(
            record.levelno == logging.INFO and "GlobalStateManager reset" in record.message
            for record in caplog.records
        )

    def test_set_tracer(self):
        """Test set_tracer sets tracer on global manager."""
        from victor.core.events import ObservabilityBus, InMemoryEventBackend

        # Create manager and tracer
        get_global_manager()
        backend = InMemoryEventBackend()
        event_bus = ObservabilityBus(backend=backend)
        tracer = StateTracer(event_bus)

        # Set tracer
        set_tracer(tracer)

        # Verify tracer is set
        assert get_tracer() == tracer

        # Verify global manager has tracer
        manager = get_global_manager()
        assert manager._tracer == tracer

    @pytest.mark.asyncio
    async def test_initialize_with_existing_workflow_state(self):
        """Test initialize_with_existing imports workflow state."""
        existing_state = {"task_id": "task-123", "status": "running"}

        manager = await initialize_with_existing(workflow_state=existing_state)

        # Verify state was imported
        assert await manager.get("task_id", scope=StateScope.WORKFLOW) == "task-123"
        assert await manager.get("status", scope=StateScope.WORKFLOW) == "running"

    @pytest.mark.asyncio
    async def test_initialize_with_existing_conversation_state(self):
        """Test initialize_with_existing imports conversation state."""
        existing_state = {"stage": "gathering", "turn_count": 5}

        manager = await initialize_with_existing(conversation_state=existing_state)

        # Verify state was imported
        assert await manager.get("stage", scope=StateScope.CONVERSATION) == "gathering"
        assert await manager.get("turn_count", scope=StateScope.CONVERSATION) == 5

    @pytest.mark.asyncio
    async def test_initialize_with_existing_team_state(self):
        """Test initialize_with_existing imports team state."""
        existing_state = {
            "coordinator": "agent-1",
            "members": ["agent-1", "agent-2"],
        }

        manager = await initialize_with_existing(team_state=existing_state)

        # Verify state was imported
        assert await manager.get("coordinator", scope=StateScope.TEAM) == "agent-1"
        assert await manager.get("members", scope=StateScope.TEAM) == [
            "agent-1",
            "agent-2",
        ]

    @pytest.mark.asyncio
    async def test_initialize_with_existing_global_state(self):
        """Test initialize_with_existing imports global state."""
        existing_state = {"debug": True, "log_level": "INFO"}

        manager = await initialize_with_existing(global_state=existing_state)

        # Verify state was imported
        assert await manager.get("debug", scope=StateScope.GLOBAL) is True
        assert await manager.get("log_level", scope=StateScope.GLOBAL) == "INFO"

    @pytest.mark.asyncio
    async def test_initialize_with_existing_all_scopes(self):
        """Test initialize_with_existing imports all scopes."""
        manager = await initialize_with_existing(
            workflow_state={"task_id": "task-123"},
            conversation_state={"stage": "gathering"},
            team_state={"coordinator": "agent-1"},
            global_state={"debug": True},
        )

        # Verify all scopes were imported
        assert await manager.get("task_id", scope=StateScope.WORKFLOW) == "task-123"
        assert await manager.get("stage", scope=StateScope.CONVERSATION) == "gathering"
        assert await manager.get("coordinator", scope=StateScope.TEAM) == "agent-1"
        assert await manager.get("debug", scope=StateScope.GLOBAL) is True


# =============================================================================
# Integration Tests
# =============================================================================


class TestGlobalStateManagerIntegration:
    """Integration tests for GlobalStateManager and factory."""

    def setup_method(self):
        """Reset global manager before each test."""
        reset_global_manager()

    def teardown_method(self):
        """Reset global manager after each test."""
        reset_global_manager()

    @pytest.mark.asyncio
    async def test_factory_with_tracer_integration(self):
        """Test factory and tracer integration."""
        from victor.core.events import ObservabilityBus, InMemoryEventBackend

        # Get manager and set tracer
        manager = get_global_manager()
        backend = InMemoryEventBackend()
        event_bus = ObservabilityBus(backend=backend)
        tracer = StateTracer(event_bus)
        set_tracer(tracer)

        # Perform operations
        await manager.set("key1", "value1", scope=StateScope.WORKFLOW)
        await manager.set("key2", "value2", scope=StateScope.CONVERSATION)

        # Verify tracer recorded both transitions
        history = tracer.get_history()
        assert len(history) == 2

    @pytest.mark.asyncio
    async def test_checkpoint_rollback_workflow(self):
        """Test checkpoint and rollback workflow."""
        manager = get_global_manager()

        # Set initial state
        await manager.set("task_id", "task-123", scope=StateScope.WORKFLOW)
        await manager.set("stage", "gathering", scope=StateScope.CONVERSATION)

        # Create checkpoint
        checkpoint = await manager.create_checkpoint()

        # Modify state
        await manager.set("task_id", "task-456", scope=StateScope.WORKFLOW)
        await manager.set("stage", "processing", scope=StateScope.CONVERSATION)

        # Verify modified state
        assert await manager.get("task_id", scope=StateScope.WORKFLOW) == "task-456"
        assert await manager.get("stage", scope=StateScope.CONVERSATION) == "processing"

        # Rollback to checkpoint
        await manager.restore_checkpoint(checkpoint)

        # Verify restored state
        assert await manager.get("task_id", scope=StateScope.WORKFLOW) == "task-123"
        assert await manager.get("stage", scope=StateScope.CONVERSATION) == "gathering"

    @pytest.mark.asyncio
    async def test_cross_scope_state_operations(self):
        """Test operations across multiple scopes."""
        manager = get_global_manager()

        # Set values in all scopes
        await manager.set("task_id", "task-123", scope=StateScope.WORKFLOW)
        await manager.set("stage", "gathering", scope=StateScope.CONVERSATION)
        await manager.set("coordinator", "agent-1", scope=StateScope.TEAM)
        await manager.set("debug", True, scope=StateScope.GLOBAL)

        # Get cross-scope state
        all_state = await manager.get_cross_scope_state()

        # Verify all scopes have data
        assert len(all_state) == 4
        assert "workflow" in all_state
        assert "conversation" in all_state
        assert "team" in all_state
        assert "global" in all_state

    @pytest.mark.asyncio
    async def test_factory_migration_scenario(self):
        """Test migration scenario using factory."""
        # Simulate legacy state data
        legacy_workflow = {"task_id": "task-123", "status": "running"}
        legacy_conversation = {"stage": "processing", "turn_count": 3}
        legacy_team = {"coordinator": "agent-1", "members": ["agent-1", "agent-2"]}

        # Migrate to new system
        manager = await initialize_with_existing(
            workflow_state=legacy_workflow,
            conversation_state=legacy_conversation,
            team_state=legacy_team,
        )

        # Verify all data migrated
        assert await manager.get("task_id", scope=StateScope.WORKFLOW) == "task-123"
        assert await manager.get("stage", scope=StateScope.CONVERSATION) == "processing"
        assert await manager.get("coordinator", scope=StateScope.TEAM) == "agent-1"

        # Create checkpoint for new system
        checkpoint = await manager.create_checkpoint()
        assert "workflow" in checkpoint
        assert "conversation" in checkpoint
        assert "team" in checkpoint


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
