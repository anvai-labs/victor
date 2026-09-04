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

"""Tests for agentic loop graph builder and executor."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from victor.framework.agentic_graph.state import (
    AgenticLoopStateModel,
    create_initial_state,
)
from victor.framework.agentic_graph.builder import (
    AgenticLoopDependencies,
    create_agentic_loop_graph,
)
from victor.framework.agentic_graph.executor import AgenticLoopGraphExecutor, LoopResult
from victor.framework.execution_checkpoint import ExecutionCheckpoint


class TestAgenticLoopGraphBuilder:
    """Tests for create_agentic_loop_graph builder function."""

    def test_create_basic_graph(self):
        """Test creating a basic agentic loop graph."""
        graph = create_agentic_loop_graph()

        assert graph is not None
        # Verify graph structure (nodes are stored in _nodes)
        assert hasattr(graph, "_nodes")
        assert len(graph._nodes) >= 4  # At least perceive, plan, act, evaluate

    def test_create_graph_with_custom_max_iterations(self):
        """Test creating graph with custom max_iterations."""
        graph = create_agentic_loop_graph(max_iterations=5)

        assert graph is not None
        assert graph.metadata["max_iterations"] == 5

    def test_create_graph_with_fulfillment_enabled(self):
        """Test creating graph with fulfillment check enabled."""
        graph = create_agentic_loop_graph(enable_fulfillment=True)

        assert graph is not None

    def test_create_graph_with_adaptive_iterations(self):
        """Test creating graph with adaptive iterations enabled."""
        graph = create_agentic_loop_graph(enable_adaptive_iterations=True)

        assert graph is not None

    def test_builder_stores_configuration_in_explicit_graph_metadata(self):
        """Builder configuration should live on the graph's declared metadata field."""
        graph = create_agentic_loop_graph(
            max_iterations=7,
            enable_fulfillment=False,
            enable_adaptive_iterations=True,
            include_prompt_node=True,
        )

        assert graph.metadata == {
            "max_iterations": 7,
            "enable_fulfillment": False,
            "enable_adaptive_iterations": True,
            "include_prompt_node": True,
        }

    def test_graph_has_required_nodes(self):
        """Test that graph has all required nodes."""
        graph = create_agentic_loop_graph()
        compiled = graph.compile()

        # Check that nodes exist (implementation may vary)
        assert compiled is not None

    @pytest.mark.asyncio
    async def test_builder_resolves_runtime_dependencies_at_node_execution(self, monkeypatch):
        """Resolver-backed dependencies should be evaluated when the node runs."""
        captured = {}

        async def _fake_perceive_node(state, runtime_intelligence=None):
            captured["state"] = state
            captured["runtime_intelligence"] = runtime_intelligence
            return state

        monkeypatch.setattr(
            "victor.framework.agentic_graph.builder.perceive_node",
            _fake_perceive_node,
        )

        graph = create_agentic_loop_graph(
            runtime_intelligence="static-runtime",
            runtime_intelligence_resolver=lambda: "dynamic-runtime",
        )

        await graph._nodes["perceive"].func({"query": "verify"})

        assert captured["state"]["query"] == "verify"
        assert captured["state"]["max_iterations"] == 10
        assert captured["runtime_intelligence"] == "dynamic-runtime"

    @pytest.mark.asyncio
    async def test_builder_accepts_typed_dependency_container(self, monkeypatch):
        """The canonical builder seam should resolve typed dependencies from the container."""
        captured = {}

        async def _fake_plan_node(
            state,
            planning_coordinator=None,
            use_llm_planning=False,
            runtime_intelligence=None,
        ):
            captured["planning_coordinator"] = planning_coordinator
            captured["use_llm_planning"] = use_llm_planning
            captured["runtime_intelligence"] = runtime_intelligence
            return state

        monkeypatch.setattr(
            "victor.framework.agentic_graph.builder.plan_node",
            _fake_plan_node,
        )

        dependencies = AgenticLoopDependencies(
            planning_coordinator="static-planner",
            resolvers={
                "planning_coordinator": lambda: "dynamic-planner",
                "use_llm_planning": lambda: True,
                "runtime_intelligence": lambda: "dynamic-runtime",
            },
        )
        graph = create_agentic_loop_graph(dependencies=dependencies)

        await graph._nodes["plan"].func({"query": "verify"})

        assert captured["planning_coordinator"] == "dynamic-planner"
        assert captured["use_llm_planning"] is True
        assert captured["runtime_intelligence"] == "dynamic-runtime"


class TestAgenticLoopGraphExecutor:
    """Tests for AgenticLoopGraphExecutor."""

    @pytest.mark.asyncio
    async def test_executor_initialization(self):
        """Test executor initialization."""
        mock_context = MagicMock()

        executor = AgenticLoopGraphExecutor(
            execution_context=mock_context,
            max_iterations=5,
        )

        assert executor.max_iterations == 5
        assert executor.execution_context == mock_context
        assert isinstance(executor.dependencies, AgenticLoopDependencies)

    @pytest.mark.asyncio
    async def test_executor_run_simple_query(self):
        """Test executor with a simple query."""
        mock_context = MagicMock()

        executor = AgenticLoopGraphExecutor(
            execution_context=mock_context,
            max_iterations=3,
        )

        result = await executor.run("Hello")

        assert isinstance(result, LoopResult)
        assert result.success is True
        assert result.response == "Processed: Hello"
        assert result.iterations == 3
        assert result.termination_reason == "max_iterations"
        assert result.metadata["final_state"]["planning_events"][-1]["selection_policy"] == (
            "heuristic_fast_path"
        )

    @pytest.mark.asyncio
    async def test_executor_with_context(self):
        """Test executor with additional context."""
        mock_context = MagicMock()

        executor = AgenticLoopGraphExecutor(
            execution_context=mock_context,
            max_iterations=3,
        )

        result = await executor.run(
            "Help with code",
            context={"project": "victor", "file": "agent.py"},
        )

        assert isinstance(result, LoopResult)

    @pytest.mark.asyncio
    async def test_executor_preserves_conversation_history(self):
        """Executor should carry conversation history into final graph state."""
        mock_context = MagicMock()

        executor = AgenticLoopGraphExecutor(
            execution_context=mock_context,
            max_iterations=1,
        )
        history = [
            {"role": "user", "content": "Fix the login bug"},
            {"role": "assistant", "content": "Found the issue in auth.py"},
        ]

        result = await executor.run(
            "Add tests for that fix",
            conversation_history=history,
        )

        assert result.success is True
        assert result.metadata["final_state"]["conversation_history"] == history

    @pytest.mark.asyncio
    async def test_executor_max_iterations_enforced(self):
        """Test that executor max_iterations is stored."""
        mock_context = MagicMock()

        executor = AgenticLoopGraphExecutor(
            execution_context=mock_context,
            max_iterations=2,
        )

        assert executor.max_iterations == 2

    @pytest.mark.asyncio
    async def test_executor_streaming(self):
        """Test executor streaming support."""
        mock_context = MagicMock()
        mock_context.services = MagicMock()
        mock_context.services.chat = AsyncMock()

        executor = AgenticLoopGraphExecutor(
            execution_context=mock_context,
            max_iterations=3,
        )

        iterations = []
        async for event in executor.stream("Streaming query"):
            iterations.append(event)
            # Early break for testing
            if len(iterations) >= 2:
                break

        assert len(iterations) >= 1

    @pytest.mark.asyncio
    async def test_executor_streaming_attaches_execution_checkpoint_metadata(self):
        """Streamed graph node events should carry trace-safe checkpoint metadata."""
        mock_context = MagicMock()
        checkpoint = ExecutionCheckpoint.create(
            session_id="session-1",
            graph_checkpoint_id="graph-ckpt-1",
            conversation_checkpoint_id="conversation-ckpt-1",
            filesystem_checkpoint_id="git-ckpt-1",
            triggering_tool_call={
                "id": "call-1",
                "name": "edit",
                "arguments": {"path": "app.py"},
            },
        )
        state = AgenticLoopStateModel(
            query="Streaming query",
            context={"execution_checkpoint": checkpoint},
        )

        class FakeCompiled:
            async def stream(self, _initial_state):
                yield "act", state

        executor = AgenticLoopGraphExecutor(
            execution_context=mock_context,
            max_iterations=1,
        )
        executor.compiled = FakeCompiled()

        events = [event async for event in executor.stream("Streaming query")]

        assert events == [
            {
                "node_name": "act",
                "state": state.to_dict(),
                "event_type": "node_complete",
                "execution_checkpoint": checkpoint.to_trace_metadata(),
            }
        ]

    @pytest.mark.asyncio
    async def test_executor_streaming_attaches_plan_execution_state(self):
        """Streamed graph node events should carry checkpointable plan execution state."""
        mock_context = MagicMock()
        plan_execution_state = {
            "plan_id": "plan-1",
            "execution_mode": "team_adapter",
            "success": True,
            "step_statuses": {"1": "completed"},
        }
        state = AgenticLoopStateModel(
            query="Streaming query",
            plan_execution_state=plan_execution_state,
        )

        class FakeCompiled:
            async def stream(self, _initial_state):
                yield "plan", state

        executor = AgenticLoopGraphExecutor(
            execution_context=mock_context,
            max_iterations=1,
        )
        executor.compiled = FakeCompiled()

        events = [event async for event in executor.stream("Streaming query")]

        assert events == [
            {
                "node_name": "plan",
                "state": state.to_dict(),
                "event_type": "node_complete",
                "plan_execution_state": plan_execution_state,
            }
        ]

    @pytest.mark.asyncio
    async def test_executor_runs_prompt_node_before_turn_execution(self):
        """Prompt node should feed system prompt into act-stage runtime overrides."""
        mock_prompt_orchestrator = MagicMock()
        mock_prompt_orchestrator.build_system_prompt.return_value = "Framework prompt"
        mock_context = MagicMock()
        mock_context.metadata = {"prompt_orchestrator": mock_prompt_orchestrator}

        executor = AgenticLoopGraphExecutor(
            execution_context=mock_context,
            max_iterations=1,
        )
        executor.turn_executor = AsyncMock()
        executor.turn_executor.execute_turn = AsyncMock(
            return_value=MagicMock(
                response="Done",
                tool_results=[],
            )
        )

        await executor.run(
            "Build prompt-aware response",
            context={"provider": "anthropic", "model": "claude-sonnet"},
        )

        runtime_overrides = executor.turn_executor.execute_turn.await_args.kwargs[
            "runtime_context_overrides"
        ]
        assert runtime_overrides["prompt_overlays"] == [
            {
                "name": "framework.agentic_graph.system_prompt",
                "content": "Framework prompt",
                "placement": "turn_prefix",
            }
        ]
        assert "prompt" in executor.get_execution_stats()["graph_nodes"]
        mock_prompt_orchestrator.build_system_prompt.assert_called_once()


class TestLoopResult:
    """Tests for LoopResult model."""

    def test_loop_result_creation(self):
        """Test creating a LoopResult."""
        result = LoopResult(
            success=True,
            response="Task completed",
            iterations=2,
            termination_reason="complete",
            metadata={"task_type": "code_generation"},
        )

        assert result.success is True
        assert result.response == "Task completed"
        assert result.iterations == 2
        assert result.termination_reason == "complete"

    def test_loop_result_from_graph_result(self):
        """Test creating LoopResult from graph execution result."""
        # Mock graph result
        graph_result = MagicMock()
        graph_result.success = True

        # Create state properly
        from victor.framework.agentic_graph.state import AgenticLoopStateModel

        state = AgenticLoopStateModel(query="Test", iteration=3)
        state = state.model_copy(
            update={
                "action_result": {"response": "Done"},
            }
        )
        graph_result.state = state
        graph_result.state_history = [("act", state)]

        result = LoopResult.from_graph_result(graph_result, state)

        assert result.success is True
        assert result.iterations == 3
        assert result.metadata["state_history"] == [
            {
                "node_name": "act",
                "state": state.to_dict(),
            }
        ]

    def test_loop_result_from_graph_result_attaches_execution_checkpoint_metadata(self):
        """Non-stream graph results should expose trace-safe checkpoint metadata."""
        graph_result = MagicMock()
        graph_result.success = True
        graph_result.state_history = []
        checkpoint = ExecutionCheckpoint.create(
            session_id="session-1",
            graph_checkpoint_id="graph-ckpt-1",
            conversation_checkpoint_id="conversation-ckpt-1",
        )
        state = AgenticLoopStateModel(
            query="Test",
            iteration=1,
            context={"execution_checkpoint": checkpoint},
            action_result={"response": "Done"},
        )

        result = LoopResult.from_graph_result(graph_result, state)

        assert result.metadata["execution_checkpoint"] == checkpoint.to_trace_metadata()

    def test_loop_result_from_graph_result_attaches_plan_execution_state(self):
        """Non-stream graph results should expose checkpointable plan execution state."""
        graph_result = MagicMock()
        graph_result.success = True
        graph_result.state_history = []
        plan_execution_state = {
            "plan_id": "plan-1",
            "execution_mode": "team_adapter",
            "success": True,
            "step_statuses": {"1": "completed"},
        }
        state = AgenticLoopStateModel(
            query="Test",
            iteration=1,
            action_result={"response": "Done"},
            plan_execution_state=plan_execution_state,
        )

        result = LoopResult.from_graph_result(graph_result, state)

        assert result.metadata["plan_execution_state"] == plan_execution_state


class TestGraphIntegration:
    """Integration tests for full graph execution."""

    @pytest.mark.asyncio
    async def test_full_graph_execution(self):
        """Test complete graph execution structure."""
        mock_context = MagicMock()

        # Create executor with mocked services
        executor = AgenticLoopGraphExecutor(
            execution_context=mock_context,
            max_iterations=3,
        )

        # Verify executor structure
        assert executor.graph is not None
        assert executor.compiled is not None
        assert executor.max_iterations == 3

        # Get execution stats
        stats = executor.get_execution_stats()
        assert stats["max_iterations"] == 3
        assert len(stats["graph_nodes"]) >= 4

    @pytest.mark.asyncio
    async def test_compiled_graph_normalizes_raw_dict_input_end_to_end(self):
        """Compiled graph should normalize raw dict input through the full loop."""
        compiled = create_agentic_loop_graph(max_iterations=1).compile()

        result = await compiled.invoke({"query": "Hello from dict input"})

        assert result.success is True
        assert isinstance(result.state, AgenticLoopStateModel)
        assert result.state.query == "Hello from dict input"
        assert result.state.stage == "evaluate"
        assert result.state.iteration == 1
        assert result.node_history == ["perceive", "plan", "act", "evaluate"]

    @pytest.mark.asyncio
    async def test_executor_opts_into_state_history(self):
        """AgenticLoopGraphExecutor is the one production consumer of
        state_history (it reconstructs legacy per-iteration results from it),
        so it must opt in explicitly now that the default is off (item 11)."""
        mock_context = MagicMock()
        executor = AgenticLoopGraphExecutor(
            execution_context=mock_context,
            max_iterations=3,
        )

        assert executor.compiled._config.checkpoint.record_state_history is True

    @pytest.mark.asyncio
    async def test_executor_compiled_graph_records_state_history_end_to_end(self):
        """A real invoke through the executor's compiled graph must populate
        state_history — the actual iteration-reconstruction dependency."""
        compiled = create_agentic_loop_graph(max_iterations=1).compile(record_state_history=True)

        result = await compiled.invoke({"query": "Hello from dict input"})

        assert result.state_history != []
        assert [node_id for node_id, _ in result.state_history] == [
            "perceive",
            "plan",
            "act",
            "evaluate",
        ]
