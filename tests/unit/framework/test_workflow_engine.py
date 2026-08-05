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

"""Tests for the framework WorkflowEngine."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.framework.workflow_engine import (
    WorkflowExecutionResult,
    WorkflowEngine,
    WorkflowEngineConfig,
    WorkflowEngineProtocol,
    WorkflowEvent,
    create_workflow_engine,
    run_yaml_workflow,
    run_graph_workflow,
)


class TestWorkflowEngineConfig:
    """Tests for WorkflowEngineConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = WorkflowEngineConfig()
        assert config.enable_caching is True
        assert config.cache_ttl_seconds == 3600
        assert config.enable_hitl is False
        assert config.hitl_timeout_seconds == 300
        assert config.enable_checkpoints is True
        assert config.max_iterations == 100
        assert config.enable_streaming is True
        assert config.parallel_execution is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = WorkflowEngineConfig(
            enable_caching=False,
            cache_ttl_seconds=1800,
            enable_hitl=True,
            hitl_timeout_seconds=600,
            enable_checkpoints=False,
            max_iterations=50,
            enable_streaming=False,
            parallel_execution=False,
        )
        assert config.enable_caching is False
        assert config.cache_ttl_seconds == 1800
        assert config.enable_hitl is True
        assert config.hitl_timeout_seconds == 600
        assert config.enable_checkpoints is False
        assert config.max_iterations == 50
        assert config.enable_streaming is False
        assert config.parallel_execution is False


class TestWorkflowExecutionResult:
    """Tests for WorkflowExecutionResult."""

    def test_successful_result(self):
        """Test creating a successful result."""
        result = WorkflowExecutionResult(
            success=True,
            final_state={"output": "data"},
            nodes_executed=["node1", "node2"],
            duration_seconds=1.5,
        )
        assert result.success is True
        assert result.final_state == {"output": "data"}
        assert result.nodes_executed == ["node1", "node2"]
        assert result.duration_seconds == 1.5
        assert result.error is None
        assert result.cached is False

    def test_failed_result(self):
        """Test creating a failed result."""
        result = WorkflowExecutionResult(
            success=False,
            error="Something went wrong",
            duration_seconds=0.5,
        )
        assert result.success is False
        assert result.error == "Something went wrong"
        assert result.final_state == {}
        assert result.nodes_executed == []

    def test_cached_result(self):
        """Test a cached result."""
        result = WorkflowExecutionResult(
            success=True,
            cached=True,
            final_state={"cached": True},
        )
        assert result.cached is True


class TestWorkflowEvent:
    """Tests for WorkflowEvent."""

    def test_basic_event(self):
        """Test creating a basic event."""
        event = WorkflowEvent(
            event_type="node_start",
            node_id="analyze",
            timestamp=1234567890.0,
        )
        assert event.event_type == "node_start"
        assert event.node_id == "analyze"
        assert event.timestamp == 1234567890.0
        assert event.data == {}
        assert event.state_snapshot is None

    def test_event_with_data(self):
        """Test event with data and state snapshot."""
        event = WorkflowEvent(
            event_type="node_complete",
            node_id="process",
            timestamp=1234567890.0,
            data={"items_processed": 10},
            state_snapshot={"count": 10, "status": "done"},
        )
        assert event.data == {"items_processed": 10}
        assert event.state_snapshot == {"count": 10, "status": "done"}


class TestWorkflowEngine:
    """Tests for WorkflowEngine class."""

    @pytest.fixture
    def engine(self):
        """Create a WorkflowEngine for testing."""
        return create_workflow_engine()

    @pytest.fixture
    def custom_engine(self):
        """Create a WorkflowEngine with custom config."""
        config = WorkflowEngineConfig(
            enable_caching=False,
            enable_hitl=True,
        )
        return create_workflow_engine(config=config)

    def test_default_creation(self, engine):
        """Test creating engine with defaults."""
        assert engine.config.enable_caching is True
        assert engine.config.enable_streaming is True

    def test_custom_config(self, custom_engine):
        """Test creating engine with custom config."""
        assert custom_engine.config.enable_caching is False
        assert custom_engine.config.enable_hitl is True

    def test_config_property(self, engine):
        """Test config property access."""
        config = engine.config
        assert isinstance(config, WorkflowEngineConfig)

    def test_enable_caching(self, engine):
        """Test enabling caching."""
        engine.enable_caching(ttl_seconds=7200)
        assert engine.config.enable_caching is True
        assert engine.config.cache_ttl_seconds == 7200

    def test_disable_caching(self, engine):
        """Test disabling caching."""
        engine.disable_caching()
        assert engine.config.enable_caching is False

    def test_get_executor_uses_runtime_executor_factory(self):
        """Legacy executor construction should flow through the shared factory seam."""
        engine = create_workflow_engine()
        sentinel = object()

        with patch(
            "victor.framework.workflow_engine.create_legacy_workflow_executor",
            return_value=sentinel,
        ) as factory:
            assert engine._get_executor() is sentinel
            assert engine._get_executor() is sentinel

        factory.assert_called_once_with()

    def test_get_streaming_executor_uses_runtime_executor_factory(self):
        """Legacy streaming executor construction should flow through the shared factory seam."""
        engine = create_workflow_engine()
        sentinel = object()

        with patch(
            "victor.framework.workflow_engine.create_legacy_streaming_workflow_executor",
            return_value=sentinel,
        ) as factory:
            assert engine._get_streaming_executor() is sentinel
            assert engine._get_streaming_executor() is sentinel

        factory.assert_called_once_with()


class TestWorkflowEngineProtocol:
    """Tests for WorkflowEngineProtocol compliance."""

    def test_protocol_compliance(self):
        """Test that WorkflowEngine satisfies the protocol."""
        engine = create_workflow_engine()
        assert isinstance(engine, WorkflowEngineProtocol)

    def test_protocol_methods_exist(self):
        """Test that all protocol methods exist."""
        engine = create_workflow_engine()
        assert hasattr(engine, "execute_yaml")
        assert hasattr(engine, "execute_graph")
        assert hasattr(engine, "stream_yaml")
        assert hasattr(engine, "stream_graph")


class TestWorkflowEngineExecution:
    """Tests for workflow execution methods."""

    @pytest.fixture
    def engine(self):
        """Create engine for execution tests."""
        return create_workflow_engine()

    @pytest.mark.asyncio
    async def test_execute_yaml_file_not_found(self, engine):
        """Test executing non-existent YAML file."""
        result = await engine.execute_yaml(
            "nonexistent_workflow.yaml",
            initial_state={},
        )
        assert result.success is False
        assert result.error is not None
        assert "No such file" in result.error or "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_graph_with_mock(self, engine):
        """Test executing a mocked CompiledGraph."""
        # Create mock graph
        mock_graph = AsyncMock()
        mock_graph.invoke = AsyncMock(return_value={"result": "success"})

        result = await engine.execute_graph(
            mock_graph,
            initial_state={"input": "test"},
        )

        assert result.success is True
        assert result.final_state == {"result": "success"}
        mock_graph.invoke.assert_called_once_with({"input": "test"})

    @pytest.mark.asyncio
    async def test_execute_graph_failure(self, engine):
        """Test graph execution failure."""
        mock_graph = AsyncMock()
        mock_graph.invoke = AsyncMock(side_effect=RuntimeError("Execution failed"))

        result = await engine.execute_graph(mock_graph, initial_state={})

        assert result.success is False
        assert "Execution failed" in result.error

    @pytest.mark.asyncio
    async def test_execute_definition_with_mock(self, engine):
        """Test executing a mocked WorkflowDefinition."""
        # Mock the executor
        with patch.object(engine, "_get_executor") as mock_get_executor:
            mock_executor = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.final_state = {"output": "done"}
            mock_result.nodes_executed = ["node1", "node2"]
            mock_result.error = None

            mock_executor.execute = AsyncMock(return_value=mock_result)
            mock_get_executor.return_value = mock_executor

            # Mock workflow definition
            mock_workflow = MagicMock()

            result = await engine.execute_definition(
                mock_workflow,
                initial_state={"input": "data"},
            )

            assert result.success is True
            assert result.final_state == {"output": "done"}
            assert result.nodes_executed == ["node1", "node2"]


class TestWorkflowEngineStreaming:
    """Tests for workflow streaming methods."""

    @pytest.fixture
    def engine(self):
        """Create engine for streaming tests."""
        return create_workflow_engine()

    @pytest.mark.asyncio
    async def test_stream_graph_with_stream_method(self, engine):
        """Test streaming a graph that has stream method."""
        # Create mock graph with stream method
        mock_graph = MagicMock()

        async def mock_stream(state):
            yield "node1", {"step": 1}
            yield "node2", {"step": 2}

        mock_graph.stream = mock_stream

        events = []
        async for event in engine.stream_graph(mock_graph, initial_state={}):
            events.append(event)

        assert len(events) == 2
        assert events[0].node_id == "node1"
        assert events[0].event_type == "node_complete"
        assert events[1].node_id == "node2"

    @pytest.mark.asyncio
    async def test_stream_graph_fallback_to_invoke(self, engine):
        """Test streaming fallback when no stream method."""
        mock_graph = AsyncMock()
        mock_graph.invoke = AsyncMock(return_value={"done": True})
        # Remove stream attribute to trigger fallback
        del mock_graph.stream

        events = []
        async for event in engine.stream_graph(mock_graph, initial_state={}):
            events.append(event)

        assert len(events) == 1
        assert events[0].event_type == "complete"
        assert events[0].state_snapshot == {"done": True}

    @pytest.mark.asyncio
    async def test_stream_graph_error_handling(self, engine):
        """Test error handling during streaming."""
        mock_graph = MagicMock()

        async def mock_stream_error(state):
            raise RuntimeError("Stream error")
            yield  # Make it a generator

        mock_graph.stream = mock_stream_error

        events = []
        async for event in engine.stream_graph(mock_graph, initial_state={}):
            events.append(event)

        assert len(events) == 1
        assert events[0].event_type == "error"
        assert "Stream error" in events[0].data.get("error", "")


class TestWorkflowEngineHITL:
    """Tests for HITL integration."""

    @pytest.fixture
    def engine(self):
        """Create engine for HITL tests."""
        config = WorkflowEngineConfig(enable_hitl=True)
        return create_workflow_engine(config=config)

    def test_set_hitl_handler(self, engine):
        """Test setting custom HITL handler."""
        mock_handler = MagicMock()
        engine.set_hitl_handler(mock_handler)
        assert engine._hitl_handler == mock_handler

    @pytest.mark.asyncio
    async def test_execute_with_hitl_file_not_found(self, engine):
        """Test HITL execution with non-existent file."""
        result = await engine.execute_with_hitl(
            "nonexistent.yaml",
            initial_state={},
        )
        assert result.success is False


class TestFactoryFunctions:
    """Tests for factory and convenience functions."""

    def test_create_workflow_engine_default(self):
        """Test creating engine with defaults."""
        engine = create_workflow_engine()
        assert isinstance(engine, WorkflowEngine)
        assert engine.config.enable_caching is True

    def test_create_workflow_engine_with_config(self):
        """Test creating engine with custom config."""
        config = WorkflowEngineConfig(enable_caching=False)
        engine = create_workflow_engine(config=config)
        assert engine.config.enable_caching is False

    @pytest.mark.asyncio
    async def test_run_yaml_workflow_convenience(self):
        """Test run_yaml_workflow convenience function."""
        # This will fail due to non-existent file, but tests the function exists
        result = await run_yaml_workflow(
            "nonexistent.yaml",
            initial_state={},
        )
        assert result.success is False

    @pytest.mark.asyncio
    async def test_run_graph_workflow_convenience(self):
        """Test run_graph_workflow convenience function."""
        mock_graph = AsyncMock()
        mock_graph.invoke = AsyncMock(return_value={"success": True})

        result = await run_graph_workflow(mock_graph, initial_state={})
        assert result.success is True


class TestWorkflowEngineCaching:
    """Tests for caching functionality."""

    @pytest.fixture
    def engine(self):
        """Create engine for caching tests."""
        return create_workflow_engine()

    def test_clear_cache_without_manager(self, engine):
        """Test clearing cache without a cache manager set."""
        # Should not raise, just do nothing
        engine.clear_cache()

    def test_cache_configuration(self, engine):
        """Test cache configuration methods."""
        # Enable with custom TTL
        engine.enable_caching(ttl_seconds=1800)
        assert engine.config.enable_caching is True
        assert engine.config.cache_ttl_seconds == 1800

        # Disable
        engine.disable_caching()
        assert engine.config.enable_caching is False
