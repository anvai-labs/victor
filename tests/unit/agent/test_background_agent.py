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

"""Tests for BackgroundAgentManager."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import time

from victor.agent.background_agent import (
    AgentStatus,
    ToolCallRecord,
    BackgroundAgent,
    BackgroundAgentManager,
    get_agent_manager,
    init_agent_manager,
)


class TestAgentStatus:
    """Tests for AgentStatus enum."""

    def test_status_values(self):
        """All status values should be strings."""
        assert AgentStatus.PENDING.value == "pending"
        assert AgentStatus.RUNNING.value == "running"
        assert AgentStatus.PAUSED.value == "paused"
        assert AgentStatus.COMPLETED.value == "completed"
        assert AgentStatus.ERROR.value == "error"
        assert AgentStatus.CANCELLED.value == "cancelled"

    def test_status_is_string_enum(self):
        """AgentStatus should be usable as string."""
        assert str(AgentStatus.RUNNING) == "AgentStatus.RUNNING"
        assert AgentStatus.COMPLETED == "completed"


class TestToolCallRecord:
    """Tests for ToolCallRecord dataclass."""

    def test_create_minimal(self):
        """Create with minimal required fields."""
        record = ToolCallRecord(
            id="tc-1",
            name="read_file",
            status="running",
            start_time=time.time(),
        )
        assert record.id == "tc-1"
        assert record.name == "read_file"
        assert record.status == "running"
        assert record.end_time is None
        assert record.arguments is None
        assert record.result is None
        assert record.error is None

    def test_create_complete(self):
        """Create with all fields."""
        start = time.time()
        end = start + 1.5
        record = ToolCallRecord(
            id="tc-2",
            name="write_file",
            status="success",
            start_time=start,
            end_time=end,
            arguments={"path": "/tmp/test.txt", "content": "hello"},
            result="File written successfully",
            error=None,
        )
        assert record.end_time == end
        assert record.arguments["path"] == "/tmp/test.txt"
        assert record.result == "File written successfully"


class TestBackgroundAgent:
    """Tests for BackgroundAgent dataclass."""

    def test_create_minimal(self):
        """Create agent with minimal fields."""
        agent = BackgroundAgent(
            id="agent-123",
            name="Test Agent",
            description="Test task",
            task="Do something",
            mode="build",
        )
        assert agent.id == "agent-123"
        assert agent.status == AgentStatus.PENDING
        assert agent.progress == 0
        assert agent.tool_calls == []
        assert agent.output is None
        assert agent.error is None

    def test_to_dict(self):
        """to_dict should serialize agent to dictionary."""
        agent = BackgroundAgent(
            id="agent-456",
            name="My Agent",
            description="A test agent",
            task="Test task",
            mode="plan",
            status=AgentStatus.RUNNING,
            progress=50,
        )
        d = agent.to_dict()
        assert d["id"] == "agent-456"
        assert d["name"] == "My Agent"
        assert d["status"] == "running"
        assert d["progress"] == 50
        assert d["mode"] == "plan"
        assert d["tool_calls"] == []

    def test_to_dict_with_tool_calls(self):
        """to_dict should include tool calls."""
        agent = BackgroundAgent(
            id="agent-789",
            name="Tool Agent",
            description="Agent with tools",
            task="Use tools",
            mode="build",
        )
        agent.tool_calls.append(
            ToolCallRecord(
                id="tc-1",
                name="bash",
                status="success",
                start_time=time.time(),
                end_time=time.time() + 0.5,
                result="Command output" * 100,  # Long result
            )
        )
        d = agent.to_dict()
        assert len(d["tool_calls"]) == 1
        assert d["tool_calls"][0]["name"] == "bash"
        # Result should be truncated to 200 chars
        assert len(d["tool_calls"][0]["result"]) <= 200

    def test_to_dict_truncates_output(self):
        """to_dict should truncate long output."""
        agent = BackgroundAgent(
            id="agent-out",
            name="Output Agent",
            description="Agent with long output",
            task="Generate output",
            mode="build",
            output="x" * 1000,  # Long output
        )
        d = agent.to_dict()
        # Output should be truncated to 500 chars
        assert len(d["output"]) <= 500


@pytest.fixture
def mock_chat_service():
    """Create a mock chat service."""
    mock = AsyncMock()

    async def mock_stream():
        yield {"type": "content", "content": "Hello"}
        yield {"type": "content", "content": " World"}

    mock.stream_chat = MagicMock(return_value=mock_stream())
    return mock


class TestBackgroundAgentManager:
    """Tests for BackgroundAgentManager."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator."""
        mock = MagicMock()
        mock.set_mode = MagicMock()

        async def mock_stream():
            yield {"type": "content", "content": "Hello"}
            yield {"type": "content", "content": " World"}

        mock.stream_chat = MagicMock(return_value=mock_stream())
        return mock

    @pytest.fixture
    def manager(self, mock_orchestrator, mock_chat_service):
        """Create a BackgroundAgentManager instance."""
        return BackgroundAgentManager(
            orchestrator=mock_orchestrator,
            chat_service=mock_chat_service,
            max_concurrent=2,
        )

    def test_init(self, manager):
        """Manager should initialize with correct defaults."""
        assert manager._max_concurrent == 2
        assert manager.active_count == 0
        assert manager._event_callback is None

    def test_set_event_callback(self, manager):
        """set_event_callback should set the callback."""
        callback = MagicMock()
        manager.set_event_callback(callback)
        assert manager._event_callback == callback

    def test_from_factory_uses_shared_sync_bridge(self, mock_chat_service):
        """from_factory should bridge async factory creation from sync code."""
        factory = MagicMock()
        orchestrator = MagicMock()
        factory.create_agent = AsyncMock(return_value=orchestrator)

        manager = BackgroundAgentManager.from_factory(
            factory, chat_service=mock_chat_service, max_concurrent=3
        )

        assert manager._orchestrator is orchestrator
        assert manager._factory is factory
        assert manager._max_concurrent == 3
        factory.create_agent.assert_awaited_once_with(mode="foreground")

    def test_from_factory_rejects_running_loop(self, mock_chat_service):
        """from_factory should fail fast when called from an async context."""
        factory = MagicMock()
        factory.create_agent = AsyncMock(return_value=MagicMock())

        async def _inner():
            with pytest.raises(RuntimeError, match="Cannot use run_sync\\(\\)"):
                BackgroundAgentManager.from_factory(factory, chat_service=mock_chat_service)

        asyncio.run(_inner())
        factory.create_agent.assert_called_once_with(mode="foreground")
        factory.create_agent.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_start_agent(self, manager):
        """start_agent should create and return agent ID."""
        agent_id = await manager.start_agent(
            task="Test task",
            mode="build",
            name="Test Agent",
        )
        assert agent_id.startswith("agent-")
        assert manager.active_count == 1

        agent = manager.get_agent(agent_id)
        assert agent is not None
        assert agent.task == "Test task"
        assert agent.mode == "build"

    @pytest.mark.asyncio
    async def test_start_agent_generates_name(self, manager):
        """start_agent should generate name from task if not provided."""
        agent_id = await manager.start_agent(
            task="A very long task description that should be truncated",
            mode="plan",
        )
        agent = manager.get_agent(agent_id)
        assert len(agent.name) <= 43  # 40 chars + "..."

    @pytest.mark.asyncio
    async def test_start_agent_max_concurrent(self, manager):
        """start_agent should raise when max concurrent reached."""
        # Start max agents
        await manager.start_agent(task="Task 1", mode="build")
        await manager.start_agent(task="Task 2", mode="build")

        # Third should fail
        with pytest.raises(RuntimeError, match="Maximum concurrent agents"):
            await manager.start_agent(task="Task 3", mode="build")

    @pytest.mark.asyncio
    async def test_cancel_agent(self, manager):
        """cancel_agent should cancel a running agent."""
        agent_id = await manager.start_agent(task="Cancel me", mode="build")

        result = await manager.cancel_agent(agent_id)
        assert result is True

        agent = manager.get_agent(agent_id)
        assert agent.status == AgentStatus.CANCELLED
        assert agent.end_time is not None

    @pytest.mark.asyncio
    async def test_cancel_agent_not_found(self, manager):
        """cancel_agent should return False for unknown agent."""
        result = await manager.cancel_agent("agent-nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_completed_agent(self, manager):
        """cancel_agent should return False for completed agent."""
        agent_id = await manager.start_agent(task="Complete me", mode="build")
        agent = manager.get_agent(agent_id)
        agent.status = AgentStatus.COMPLETED

        result = await manager.cancel_agent(agent_id)
        assert result is False

    def test_get_agent(self, manager):
        """get_agent should return None for unknown agent."""
        assert manager.get_agent("nonexistent") is None

    def test_get_agent_status(self, manager):
        """get_agent_status should return None for unknown agent."""
        assert manager.get_agent_status("nonexistent") is None

    @pytest.mark.asyncio
    async def test_get_agent_status_dict(self, manager):
        """get_agent_status should return dict for existing agent."""
        agent_id = await manager.start_agent(task="Status test", mode="build")
        status = manager.get_agent_status(agent_id)

        assert status is not None
        assert status["id"] == agent_id
        assert status["task"] == "Status test"

    @pytest.mark.asyncio
    async def test_list_agents(self, manager):
        """list_agents should return all agents."""
        await manager.start_agent(task="Task 1", mode="build")
        await manager.start_agent(task="Task 2", mode="plan")

        agents = manager.list_agents()
        assert len(agents) == 2

    @pytest.mark.asyncio
    async def test_list_agents_filter_status(self, manager):
        """list_agents should filter by status."""
        agent_id = await manager.start_agent(task="Task 1", mode="build")
        agent = manager.get_agent(agent_id)
        agent.status = AgentStatus.COMPLETED

        running = manager.list_agents(status=AgentStatus.RUNNING)
        completed = manager.list_agents(status=AgentStatus.COMPLETED)

        # The second agent is still running/pending
        assert len(completed) == 1

    @pytest.mark.asyncio
    async def test_list_agents_limit(self, manager):
        """list_agents should respect limit."""
        # Need to increase max_concurrent for this test
        manager._max_concurrent = 10
        for i in range(5):
            await manager.start_agent(task=f"Task {i}", mode="build")

        agents = manager.list_agents(limit=3)
        assert len(agents) == 3

    @pytest.mark.asyncio
    async def test_clear_completed(self, manager):
        """clear_completed should remove completed agents."""
        agent_id = await manager.start_agent(task="Complete me", mode="build")
        agent = manager.get_agent(agent_id)
        agent.status = AgentStatus.COMPLETED

        cleared = manager.clear_completed()
        assert cleared == 1
        assert manager.get_agent(agent_id) is None

    def test_emit_event_with_callback(self, manager):
        """_emit_event should call callback with data."""
        callback = MagicMock()
        manager.set_event_callback(callback)

        manager._emit_event("test_event", {"key": "value"})

        callback.assert_called_once_with("test_event", {"key": "value"})

    def test_emit_event_without_callback(self, manager):
        """_emit_event should not fail without callback."""
        # Should not raise
        manager._emit_event("test_event", {"key": "value"})

    def test_emit_event_callback_error(self, manager):
        """_emit_event should handle callback errors gracefully."""
        callback = MagicMock(side_effect=Exception("Callback error"))
        manager.set_event_callback(callback)

        # Should not raise
        manager._emit_event("test_event", {"key": "value"})


class TestGlobalFunctions:
    """Tests for module-level functions."""

    def test_get_agent_manager_uninitialized(self):
        """get_agent_manager should return None before initialization."""
        # Reset global state by accessing the already-imported module from sys.modules
        # to avoid triggering a fresh import lookup that can fail with mirrored test dirs
        import sys

        module = sys.modules["victor.agent.background_agent"]
        module._agent_manager = None
        assert get_agent_manager() is None

    def test_init_agent_manager(self, mock_chat_service):
        """init_agent_manager should create and return manager."""
        mock_orch = MagicMock()

        manager = init_agent_manager(
            orchestrator=mock_orch,
            chat_service=mock_chat_service,
            max_concurrent=8,
        )

        assert manager is not None
        assert manager._max_concurrent == 8
        assert get_agent_manager() is manager

    def test_init_agent_manager_with_callback(self, mock_chat_service):
        """init_agent_manager should accept event callback."""
        mock_orch = MagicMock()
        callback = MagicMock()

        manager = init_agent_manager(
            orchestrator=mock_orch,
            chat_service=mock_chat_service,
            event_callback=callback,
        )

        assert manager._event_callback is callback
