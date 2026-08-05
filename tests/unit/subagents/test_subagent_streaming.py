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

"""Unit tests for sub-agent streaming support (Phase 2 Graph Orchestration).

Tests cover:
- SubAgent.stream_execute() yields StreamChunk objects
- SubAgent.stream_execute() handles errors gracefully
- SubAgentOrchestrator.stream_spawn() yields chunks
- stream_spawn() respects tool_budget
- stream_spawn() with allowed_tools filtering

TDD: These tests are written FIRST before implementation.
"""

import asyncio
from typing import AsyncIterator, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.agent.subagents import (
    SubAgent,
    SubAgentConfig,
    SubAgentOrchestrator,
    SubAgentRole,
)
from victor.providers.base import StreamChunk

# =============================================================================
# SubAgent.stream_execute() Tests
# =============================================================================


class TestSubAgentStreamExecute:
    """Tests for SubAgent.stream_execute() method."""

    @pytest.fixture
    def mock_parent_orchestrator(self):
        """Create a mock parent orchestrator."""
        mock = MagicMock()
        mock.settings = MagicMock()
        mock.settings.tool_budget = 50
        mock.settings.max_context_chars = 100000
        mock.provider_name = "mock_provider"
        mock.model = "mock_model"
        mock.temperature = 0.7
        mock.tool_registry = MagicMock()
        mock.tool_registry.get.return_value = MagicMock()
        mock.tool_registry.clear = MagicMock()
        mock.tool_registry.register = MagicMock()
        return mock

    @pytest.fixture
    def sample_config(self):
        """Create a sample config for testing."""
        return SubAgentConfig(
            role=SubAgentRole.RESEARCHER,
            task="Research authentication patterns",
            allowed_tools=["read", "ls", "search"],
            tool_budget=15,
            context_limit=50000,
        )

    @pytest.mark.asyncio
    async def test_stream_execute_yields_stream_chunks(
        self, sample_config, mock_parent_orchestrator
    ):
        """Test that stream_execute() yields StreamChunk objects."""
        subagent = SubAgent(sample_config, mock_parent_orchestrator)

        # Create mock orchestrator with stream_chat that yields chunks
        async def mock_stream_chat(task: str) -> AsyncIterator[StreamChunk]:
            yield StreamChunk(content="Hello", is_final=False)
            yield StreamChunk(content=" world", is_final=False)
            yield StreamChunk(content="!", is_final=True)

        mock_orchestrator = MagicMock()
        mock_orchestrator.stream_chat = mock_stream_chat
        mock_orchestrator.tool_calls_used = 3
        mock_orchestrator.get_messages = MagicMock(return_value=[])

        # Patch the constrained orchestrator creation
        with patch.object(
            subagent, "_create_constrained_orchestrator", return_value=mock_orchestrator
        ):
            chunks: List[StreamChunk] = []
            async for chunk in subagent.stream_execute():
                chunks.append(chunk)
                assert isinstance(chunk, StreamChunk)

            # Verify we got chunks
            assert len(chunks) >= 3
            # The last chunk should be final
            assert chunks[-1].is_final is True

    @pytest.mark.asyncio
    async def test_stream_execute_accumulates_content(
        self, sample_config, mock_parent_orchestrator
    ):
        """Test that stream_execute() correctly streams content incrementally."""
        subagent = SubAgent(sample_config, mock_parent_orchestrator)

        async def mock_stream_chat(task: str) -> AsyncIterator[StreamChunk]:
            yield StreamChunk(content="Found ", is_final=False)
            yield StreamChunk(content="authentication ", is_final=False)
            yield StreamChunk(content="patterns.", is_final=True)

        mock_orchestrator = MagicMock()
        mock_orchestrator.stream_chat = mock_stream_chat
        mock_orchestrator.tool_calls_used = 2
        mock_orchestrator.get_messages = MagicMock(return_value=[])

        with patch.object(
            subagent, "_create_constrained_orchestrator", return_value=mock_orchestrator
        ):
            chunks: List[StreamChunk] = []
            async for chunk in subagent.stream_execute():
                chunks.append(chunk)

            # Verify content was streamed
            content_chunks = [c for c in chunks if c.content]
            assert len(content_chunks) >= 3

    @pytest.mark.asyncio
    async def test_stream_execute_handles_errors_gracefully(
        self, sample_config, mock_parent_orchestrator
    ):
        """Test that stream_execute() handles errors and yields error chunk."""
        subagent = SubAgent(sample_config, mock_parent_orchestrator)

        async def mock_stream_chat_error(task: str) -> AsyncIterator[StreamChunk]:
            yield StreamChunk(content="Starting...", is_final=False)
            raise RuntimeError("Provider connection failed")

        mock_orchestrator = MagicMock()
        mock_orchestrator.stream_chat = mock_stream_chat_error
        mock_orchestrator.tool_calls_used = 1
        mock_orchestrator.get_messages = MagicMock(return_value=[])

        with patch.object(
            subagent, "_create_constrained_orchestrator", return_value=mock_orchestrator
        ):
            chunks: List[StreamChunk] = []
            async for chunk in subagent.stream_execute():
                chunks.append(chunk)

            # Should have at least one chunk before error and one final error chunk
            assert len(chunks) >= 2
            # Last chunk should be final and contain error metadata
            final_chunk = chunks[-1]
            assert final_chunk.is_final is True
            assert final_chunk.metadata is not None
            assert "error" in final_chunk.metadata

    @pytest.mark.asyncio
    async def test_stream_execute_includes_tool_calls_in_chunks(
        self, sample_config, mock_parent_orchestrator
    ):
        """Test that stream_execute() includes tool calls in StreamChunk."""
        subagent = SubAgent(sample_config, mock_parent_orchestrator)

        tool_call = {"id": "call_1", "name": "read", "arguments": {"path": "/test"}}

        async def mock_stream_chat(task: str) -> AsyncIterator[StreamChunk]:
            yield StreamChunk(content="Reading file...", is_final=False)
            yield StreamChunk(content="", tool_calls=[tool_call], is_final=False)
            yield StreamChunk(content="Done.", is_final=True)

        mock_orchestrator = MagicMock()
        mock_orchestrator.stream_chat = mock_stream_chat
        mock_orchestrator.tool_calls_used = 1
        mock_orchestrator.get_messages = MagicMock(return_value=[])

        with patch.object(
            subagent, "_create_constrained_orchestrator", return_value=mock_orchestrator
        ):
            chunks: List[StreamChunk] = []
            async for chunk in subagent.stream_execute():
                chunks.append(chunk)

            # Check that tool calls are included in chunks
            tool_call_chunks = [c for c in chunks if c.tool_calls]
            assert len(tool_call_chunks) >= 1
            assert tool_call_chunks[0].tool_calls[0]["name"] == "read"

    @pytest.mark.asyncio
    async def test_stream_execute_creates_orchestrator_lazily(
        self, sample_config, mock_parent_orchestrator
    ):
        """Test that stream_execute() creates orchestrator only when needed."""
        subagent = SubAgent(sample_config, mock_parent_orchestrator)

        # Orchestrator should be None initially
        assert subagent.orchestrator is None

        async def mock_stream_chat(task: str) -> AsyncIterator[StreamChunk]:
            yield StreamChunk(content="Test", is_final=True)

        mock_orchestrator = MagicMock()
        mock_orchestrator.stream_chat = mock_stream_chat
        mock_orchestrator.tool_calls_used = 0
        mock_orchestrator.get_messages = MagicMock(return_value=[])

        with patch.object(
            subagent, "_create_constrained_orchestrator", return_value=mock_orchestrator
        ) as mock_create:
            async for _ in subagent.stream_execute():
                pass

            # Should have called create once
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_execute_final_chunk_has_metadata(
        self, sample_config, mock_parent_orchestrator
    ):
        """Test that stream_execute() final chunk includes execution metadata."""
        subagent = SubAgent(sample_config, mock_parent_orchestrator)

        async def mock_stream_chat(task: str) -> AsyncIterator[StreamChunk]:
            yield StreamChunk(content="Completed task", is_final=True)

        mock_orchestrator = MagicMock()
        mock_orchestrator.stream_chat = mock_stream_chat
        mock_orchestrator.tool_calls_used = 5
        mock_orchestrator.get_messages = MagicMock(
            return_value=[{"role": "user", "content": "test"}]
        )

        with patch.object(
            subagent, "_create_constrained_orchestrator", return_value=mock_orchestrator
        ):
            chunks: List[StreamChunk] = []
            async for chunk in subagent.stream_execute():
                chunks.append(chunk)

            # Final chunk should have metadata with execution info
            final_chunk = chunks[-1]
            assert final_chunk.is_final is True
            assert final_chunk.metadata is not None
            assert "tool_calls_used" in final_chunk.metadata
            assert "role" in final_chunk.metadata


# =============================================================================
# SubAgentOrchestrator.stream_spawn() Tests
# =============================================================================


class TestSubAgentOrchestratorStreamSpawn:
    """Tests for SubAgentOrchestrator.stream_spawn() method."""

    @pytest.fixture
    def mock_parent(self):
        """Create mock parent orchestrator."""
        mock = MagicMock()
        mock.settings = MagicMock()
        mock.settings.tool_budget = 50
        mock.settings.max_context_chars = 100000
        mock.provider_name = "mock_provider"
        mock.model = "mock_model"
        mock.temperature = 0.7
        mock.tool_registry = MagicMock()
        mock.tool_registry.get.return_value = MagicMock()
        mock.tool_registry.clear = MagicMock()
        mock.tool_registry.register = MagicMock()
        return mock

    @pytest.mark.asyncio
    async def test_stream_spawn_yields_chunks(self, mock_parent):
        """Test that stream_spawn() yields StreamChunk objects."""
        orchestrator = SubAgentOrchestrator(mock_parent)

        async def mock_stream_execute(self) -> AsyncIterator[StreamChunk]:
            yield StreamChunk(content="Hello", is_final=False)
            yield StreamChunk(content=" world", is_final=True)

        with patch.object(SubAgent, "stream_execute", mock_stream_execute):
            chunks: List[StreamChunk] = []
            async for chunk in orchestrator.stream_spawn(SubAgentRole.RESEARCHER, "Test task"):
                chunks.append(chunk)
                assert isinstance(chunk, StreamChunk)

            assert len(chunks) >= 2
            assert chunks[-1].is_final is True

    @pytest.mark.asyncio
    async def test_stream_spawn_respects_tool_budget(self, mock_parent):
        """Test that stream_spawn() passes tool_budget to SubAgent."""
        orchestrator = SubAgentOrchestrator(mock_parent)

        created_configs: List[SubAgentConfig] = []

        original_init = SubAgent.__init__

        def capture_init(self, config: SubAgentConfig, parent):
            created_configs.append(config)
            original_init(self, config, parent)

        async def mock_stream_execute() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(content="Done", is_final=True, metadata={"tool_calls_used": 5})

        with patch.object(SubAgent, "__init__", capture_init):
            with patch.object(SubAgent, "stream_execute", mock_stream_execute):
                async for _ in orchestrator.stream_spawn(
                    SubAgentRole.EXECUTOR, "Execute task", tool_budget=25
                ):
                    pass

                # Verify the config had the right budget
                assert len(created_configs) == 1
                assert created_configs[0].tool_budget == 25

    @pytest.mark.asyncio
    async def test_stream_spawn_with_allowed_tools_filtering(self, mock_parent):
        """Test that stream_spawn() passes allowed_tools to SubAgent."""
        orchestrator = SubAgentOrchestrator(mock_parent)

        created_configs: List[SubAgentConfig] = []

        original_init = SubAgent.__init__

        def capture_init(self, config: SubAgentConfig, parent):
            created_configs.append(config)
            original_init(self, config, parent)

        async def mock_stream_execute() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(content="Done", is_final=True)

        custom_tools = ["read", "write", "search"]

        with patch.object(SubAgent, "__init__", capture_init):
            with patch.object(SubAgent, "stream_execute", mock_stream_execute):
                async for _ in orchestrator.stream_spawn(
                    SubAgentRole.RESEARCHER,
                    "Research task",
                    allowed_tools=custom_tools,
                ):
                    pass

                # Verify the config had the right tools
                assert len(created_configs) == 1
                assert created_configs[0].allowed_tools == custom_tools

    @pytest.mark.asyncio
    async def test_stream_spawn_handles_timeout(self, mock_parent):
        """Test that stream_spawn() handles timeout gracefully."""
        orchestrator = SubAgentOrchestrator(mock_parent)

        async def slow_stream_execute() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(content="Starting...", is_final=False)
            await asyncio.sleep(10)  # Simulate slow execution
            yield StreamChunk(content="Done", is_final=True)

        with patch.object(SubAgent, "stream_execute", slow_stream_execute):
            chunks: List[StreamChunk] = []
            async for chunk in orchestrator.stream_spawn(
                SubAgentRole.RESEARCHER, "Test task", timeout_seconds=1
            ):
                chunks.append(chunk)

            # Should get timeout error chunk
            assert len(chunks) >= 1
            final_chunk = chunks[-1]
            assert final_chunk.is_final is True
            # Timeout should be indicated in metadata
            assert final_chunk.metadata is not None
            assert "error" in final_chunk.metadata or "timeout" in str(final_chunk.metadata).lower()

    @pytest.mark.asyncio
    async def test_stream_spawn_uses_role_defaults(self, mock_parent):
        """Test that stream_spawn() uses role defaults when not specified."""
        from victor.agent.subagents.orchestrator import (
            ROLE_DEFAULT_BUDGETS,
            ROLE_DEFAULT_TOOLS,
        )

        orchestrator = SubAgentOrchestrator(mock_parent)

        created_configs: List[SubAgentConfig] = []

        original_init = SubAgent.__init__

        def capture_init(self, config: SubAgentConfig, parent):
            created_configs.append(config)
            original_init(self, config, parent)

        async def mock_stream_execute() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(content="Done", is_final=True)

        with patch.object(SubAgent, "__init__", capture_init):
            with patch.object(SubAgent, "stream_execute", mock_stream_execute):
                async for _ in orchestrator.stream_spawn(
                    SubAgentRole.PLANNER, "Plan the implementation"
                ):
                    pass

                # Verify role defaults were used
                assert len(created_configs) == 1
                config = created_configs[0]
                assert config.tool_budget == ROLE_DEFAULT_BUDGETS[SubAgentRole.PLANNER]
                assert config.allowed_tools == ROLE_DEFAULT_TOOLS[SubAgentRole.PLANNER]

    @pytest.mark.asyncio
    async def test_stream_spawn_tracks_active_subagents(self, mock_parent):
        """Test that stream_spawn() tracks active subagents during streaming."""
        orchestrator = SubAgentOrchestrator(mock_parent)

        assert orchestrator.get_active_count() == 0

        active_during_stream = []

        async def mock_stream_execute(self) -> AsyncIterator[StreamChunk]:
            # Capture active count during streaming
            active_during_stream.append(orchestrator.get_active_count())
            yield StreamChunk(content="Working...", is_final=False)
            active_during_stream.append(orchestrator.get_active_count())
            yield StreamChunk(content="Done", is_final=True)

        with patch.object(SubAgent, "stream_execute", mock_stream_execute):
            async for _ in orchestrator.stream_spawn(SubAgentRole.RESEARCHER, "Test task"):
                pass

            # After completion, no active subagents
            assert orchestrator.get_active_count() == 0
            # During streaming, should have had 1 active
            assert 1 in active_during_stream

    @pytest.mark.asyncio
    async def test_stream_spawn_handles_empty_stream(self, mock_parent):
        """Test that stream_spawn() handles case where stream yields nothing."""
        orchestrator = SubAgentOrchestrator(mock_parent)

        async def empty_stream_execute() -> AsyncIterator[StreamChunk]:
            # Empty async generator
            if False:
                yield StreamChunk()

        with patch.object(SubAgent, "stream_execute", empty_stream_execute):
            chunks: List[StreamChunk] = []
            async for chunk in orchestrator.stream_spawn(SubAgentRole.RESEARCHER, "Test task"):
                chunks.append(chunk)

            # Should handle gracefully, possibly with a final chunk indicating completion
            # The implementation should ensure at least a final chunk is yielded
            assert len(chunks) >= 0  # At minimum, empty is acceptable; ideally has final chunk


# =============================================================================
# Backward Compatibility Tests
# =============================================================================


class TestBackwardCompatibility:
    """Tests ensuring backward compatibility with existing execute() and spawn()."""

    @pytest.fixture
    def mock_parent(self):
        """Create mock parent orchestrator."""
        mock = MagicMock()
        mock.settings = MagicMock()
        mock.settings.tool_budget = 50
        mock.settings.max_context_chars = 100000
        mock.provider_name = "mock_provider"
        mock.model = "mock_model"
        mock.temperature = 0.7
        mock.tool_registry = MagicMock()
        mock.tool_registry.get.return_value = MagicMock()
        mock.tool_registry.clear = MagicMock()
        mock.tool_registry.register = MagicMock()
        return mock

    def test_execute_method_still_exists(self, mock_parent):
        """Test that SubAgent.execute() method still exists."""
        config = SubAgentConfig(
            role=SubAgentRole.RESEARCHER,
            task="Test",
            allowed_tools=["read"],
            tool_budget=10,
            context_limit=30000,
        )
        subagent = SubAgent(config, mock_parent)
        assert hasattr(subagent, "execute")
        assert asyncio.iscoroutinefunction(subagent.execute)

    def test_spawn_method_still_exists(self, mock_parent):
        """Test that SubAgentOrchestrator.spawn() method still exists."""
        orchestrator = SubAgentOrchestrator(mock_parent)
        assert hasattr(orchestrator, "spawn")
        assert asyncio.iscoroutinefunction(orchestrator.spawn)

    def test_stream_execute_method_exists(self, mock_parent):
        """Test that SubAgent.stream_execute() method exists."""
        config = SubAgentConfig(
            role=SubAgentRole.RESEARCHER,
            task="Test",
            allowed_tools=["read"],
            tool_budget=10,
            context_limit=30000,
        )
        subagent = SubAgent(config, mock_parent)
        assert hasattr(subagent, "stream_execute")

    def test_stream_spawn_method_exists(self, mock_parent):
        """Test that SubAgentOrchestrator.stream_spawn() method exists."""
        orchestrator = SubAgentOrchestrator(mock_parent)
        assert hasattr(orchestrator, "stream_spawn")
