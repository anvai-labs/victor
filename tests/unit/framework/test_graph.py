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

"""Tests for victor.framework.graph module (StateGraph).

These tests verify the LangGraph-compatible StateGraph implementation
including cyclic workflows, checkpointing, and typed state management.
"""

import logging
import pytest
from typing import TypedDict, Optional, List
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import BaseModel

from victor.framework.graph import (
    StateGraph,
    CompiledGraph,
    Node,
    Edge,
    EdgeType,
    FrameworkNodeStatus,
    GraphExecutionResult,
    GraphConfig,
    GraphEventEmitter,
    WorkflowCheckpoint,
    CheckpointerProtocol,
    MemoryCheckpointer,
    RLCheckpointerAdapter,
    SubgraphNode,
    Send,
    default_state_merger,
    ParallelBranchExecutionError,
    strict_state_merger,
    END,
    START,
    create_graph,
)

from victor.framework.config import (
    ExecutionConfig,
    CheckpointConfig,
    InterruptConfig,
    PerformanceConfig,
    ObservabilityConfig,
)
from victor.framework.execution_checkpoint import ExecutionCheckpoint

# =============================================================================
# Test State Types
# =============================================================================


class SimpleState(TypedDict):
    """Simple state for testing."""

    value: int
    history: List[str]


class TaskState(TypedDict, total=False):
    """Task state with optional fields."""

    task: str
    result: Optional[str]
    iteration: int
    complete: bool


class PydanticCounterState(BaseModel):
    """Pydantic state used to verify copy-on-write preserves returned models."""

    iteration: int = 0


# =============================================================================
# Node Functions for Testing
# =============================================================================


async def increment_node(state: SimpleState) -> SimpleState:
    """Node that increments value."""
    state["value"] += 1
    state["history"].append("increment")
    return state


async def double_node(state: SimpleState) -> SimpleState:
    """Node that doubles value."""
    state["value"] *= 2
    state["history"].append("double")
    return state


def sync_node(state: SimpleState) -> SimpleState:
    """Synchronous node for testing."""
    state["value"] += 10
    state["history"].append("sync")
    return state


async def task_process_node(state: TaskState) -> TaskState:
    """Process task and increment iteration."""
    state["iteration"] = state.get("iteration", 0) + 1
    return state


async def task_check_node(state: TaskState) -> TaskState:
    """Check if task is complete."""
    if state.get("iteration", 0) >= 3:
        state["complete"] = True
        state["result"] = "Completed"
    return state


def should_continue(state: TaskState) -> str:
    """Condition: continue processing or end."""
    if state.get("complete", False):
        return "done"
    return "continue"


# =============================================================================
# Graph Event Tests
# =============================================================================


class TestGraphEventEmitter:
    def test_emit_graph_completed_unifies_lifecycle_and_rl_events(self):
        """Graph completion should emit both lifecycle and RL completion signals once."""
        mock_bus = MagicMock()
        mock_hooks = MagicMock()

        with (
            patch("victor.core.events.get_observability_bus", return_value=mock_bus),
            patch("victor.framework.rl.hooks.get_rl_hooks", return_value=mock_hooks),
        ):
            emitter = GraphEventEmitter(graph_id="graph-1", emit_events=True)
            emitter.emit_graph_completed(
                success=True,
                iterations=2,
                duration=1.5,
                node_count=2,
            )

        mock_bus.emit_lifecycle_event.assert_called_once()
        event_name, payload = mock_bus.emit_lifecycle_event.call_args.args
        assert event_name == "graph_completed"
        assert payload["graph_id"] == "graph-1"
        assert payload["success"] is True

        mock_hooks.emit.assert_called_once()
        rl_event = mock_hooks.emit.call_args.args[0]
        assert rl_event.workflow_name == "state_graph"
        assert rl_event.success is True
        assert rl_event.metadata["iterations"] == 2


# =============================================================================
# Node and Edge Tests
# =============================================================================


class TestNode:
    """Tests for Node class."""

    def test_node_creation(self):
        """Node should store id, func, and metadata."""
        node = Node(id="test", func=increment_node)

        assert node.id == "test"
        assert node.func == increment_node
        assert node.metadata == {}

    def test_node_with_metadata(self):
        """Node should store metadata."""
        node = Node(id="test", func=increment_node, metadata={"key": "value"})

        assert node.metadata == {"key": "value"}

    @pytest.mark.asyncio
    async def test_node_execute_async(self):
        """execute should run async function."""
        node = Node(id="inc", func=increment_node)
        state: SimpleState = {"value": 5, "history": []}

        result = await node.execute(state)

        assert result["value"] == 6
        assert result["history"] == ["increment"]

    @pytest.mark.asyncio
    async def test_node_execute_sync(self):
        """execute should handle sync function."""
        node = Node(id="sync", func=sync_node)
        state: SimpleState = {"value": 5, "history": []}

        result = await node.execute(state)

        assert result["value"] == 15
        assert result["history"] == ["sync"]


class TestEdge:
    """Tests for Edge class."""

    def test_normal_edge(self):
        """Normal edge should store source and target."""
        edge = Edge(source="a", target="b")

        assert edge.source == "a"
        assert edge.target == "b"
        assert edge.edge_type == EdgeType.NORMAL
        assert edge.condition is None

    def test_conditional_edge(self):
        """Conditional edge should store branches and condition."""
        edge = Edge(
            source="a",
            target={"yes": "b", "no": "c"},
            edge_type=EdgeType.CONDITIONAL,
            condition=lambda s: "yes" if s.get("flag") else "no",
        )

        assert edge.edge_type == EdgeType.CONDITIONAL
        assert edge.target == {"yes": "b", "no": "c"}

    def test_get_target_normal(self):
        """get_target should return target for normal edge."""
        edge = Edge(source="a", target="b")

        assert edge.get_target({}) == "b"

    def test_get_target_conditional(self):
        """get_target should evaluate condition for conditional edge."""
        edge = Edge(
            source="a",
            target={"yes": "b", "no": "c"},
            edge_type=EdgeType.CONDITIONAL,
            condition=lambda s: "yes" if s.get("flag") else "no",
        )

        assert edge.get_target({"flag": True}) == "b"
        assert edge.get_target({"flag": False}) == "c"


# =============================================================================
# WorkflowCheckpoint Tests
# =============================================================================


class TestCheckpoint:
    """Tests for WorkflowCheckpoint class."""

    def test_checkpoint_creation(self):
        """WorkflowCheckpoint should store all fields."""
        checkpoint = WorkflowCheckpoint(
            checkpoint_id="cp1",
            thread_id="t1",
            node_id="node1",
            state={"value": 42},
            timestamp=1234567890.0,
            metadata={"key": "value"},
        )

        assert checkpoint.checkpoint_id == "cp1"
        assert checkpoint.thread_id == "t1"
        assert checkpoint.node_id == "node1"
        assert checkpoint.state == {"value": 42}
        assert checkpoint.timestamp == 1234567890.0
        assert checkpoint.metadata == {"key": "value"}

    def test_to_dict(self):
        """to_dict should serialize checkpoint."""
        checkpoint = WorkflowCheckpoint(
            checkpoint_id="cp1",
            thread_id="t1",
            node_id="node1",
            state={"value": 42},
            timestamp=1234567890.0,
        )

        d = checkpoint.to_dict()

        assert d["checkpoint_id"] == "cp1"
        assert d["thread_id"] == "t1"
        assert d["state"] == {"value": 42}

    def test_from_dict(self):
        """from_dict should deserialize checkpoint."""
        data = {
            "checkpoint_id": "cp1",
            "thread_id": "t1",
            "node_id": "node1",
            "state": {"value": 42},
            "timestamp": 1234567890.0,
            "metadata": {"key": "value"},
        }

        checkpoint = WorkflowCheckpoint.from_dict(data)

        assert checkpoint.checkpoint_id == "cp1"
        assert checkpoint.metadata == {"key": "value"}


class TestMemoryCheckpointer:
    """Tests for MemoryCheckpointer."""

    @pytest.mark.asyncio
    async def test_save_and_load(self):
        """save and load should work together."""
        checkpointer = MemoryCheckpointer()

        checkpoint = WorkflowCheckpoint(
            checkpoint_id="cp1",
            thread_id="t1",
            node_id="node1",
            state={"value": 42},
            timestamp=1234567890.0,
        )

        await checkpointer.save(checkpoint)
        loaded = await checkpointer.load("t1")

        assert loaded is not None
        assert loaded.checkpoint_id == "cp1"
        assert loaded.state == {"value": 42}

    @pytest.mark.asyncio
    async def test_load_returns_latest(self):
        """load should return latest checkpoint."""
        checkpointer = MemoryCheckpointer()

        await checkpointer.save(
            WorkflowCheckpoint(
                checkpoint_id="cp1",
                thread_id="t1",
                node_id="node1",
                state={"value": 1},
                timestamp=1.0,
            )
        )
        await checkpointer.save(
            WorkflowCheckpoint(
                checkpoint_id="cp2",
                thread_id="t1",
                node_id="node2",
                state={"value": 2},
                timestamp=2.0,
            )
        )

        loaded = await checkpointer.load("t1")

        assert loaded.checkpoint_id == "cp2"
        assert loaded.state == {"value": 2}

    @pytest.mark.asyncio
    async def test_load_unknown_thread(self):
        """load should return None for unknown thread."""
        checkpointer = MemoryCheckpointer()

        loaded = await checkpointer.load("unknown")

        assert loaded is None

    @pytest.mark.asyncio
    async def test_list_checkpoints(self):
        """list should return all checkpoints for thread."""
        checkpointer = MemoryCheckpointer()

        await checkpointer.save(
            WorkflowCheckpoint(
                checkpoint_id="cp1",
                thread_id="t1",
                node_id="n1",
                state={},
                timestamp=1.0,
            )
        )
        await checkpointer.save(
            WorkflowCheckpoint(
                checkpoint_id="cp2",
                thread_id="t1",
                node_id="n2",
                state={},
                timestamp=2.0,
            )
        )

        checkpoints = await checkpointer.list("t1")

        assert len(checkpoints) == 2


# =============================================================================
# StateGraph Tests
# =============================================================================


class TestStateGraph:
    """Tests for StateGraph class."""

    def test_create_graph(self):
        """StateGraph should be creatable."""
        graph = StateGraph(SimpleState)

        assert graph._state_schema == SimpleState
        assert graph._nodes == {}
        assert graph._edges == {}

    def test_create_graph_with_metadata(self):
        """StateGraph should expose explicit graph-level metadata."""
        graph = StateGraph(SimpleState, metadata={"owner": "agentic-loop", "version": 1})

        assert graph.metadata == {"owner": "agentic-loop", "version": 1}

    def test_add_node(self):
        """add_node should register node."""
        graph = StateGraph(SimpleState)

        graph.add_node("inc", increment_node)

        assert "inc" in graph._nodes
        assert graph._nodes["inc"].func == increment_node

    def test_add_node_chaining(self):
        """add_node should return self for chaining."""
        graph = StateGraph(SimpleState)

        result = graph.add_node("a", increment_node).add_node("b", double_node)

        assert result is graph
        assert len(graph._nodes) == 2

    def test_node_ids_reflect_registered_nodes(self):
        """node_ids should expose registered nodes without reaching into internals."""
        graph = StateGraph(SimpleState)

        graph.add_node("a", increment_node).add_node("b", double_node)

        assert graph.node_ids == ["a", "b"]

    def test_add_duplicate_node_raises(self):
        """add_node should raise for duplicate nodes."""
        graph = StateGraph(SimpleState)
        graph.add_node("a", increment_node)

        with pytest.raises(ValueError, match="already exists"):
            graph.add_node("a", double_node)

    def test_add_edge(self):
        """add_edge should register edge."""
        graph = StateGraph(SimpleState)
        graph.add_node("a", increment_node)
        graph.add_node("b", double_node)

        graph.add_edge("a", "b")

        assert len(graph._edges["a"]) == 1
        assert graph._edges["a"][0].target == "b"

    def test_add_conditional_edge(self):
        """add_conditional_edge should register conditional edge."""
        graph = StateGraph(TaskState)
        graph.add_node("a", task_process_node)
        graph.add_node("b", task_check_node)

        graph.add_conditional_edge("a", should_continue, {"continue": "a", "done": "b"})

        edge = graph._edges["a"][0]
        assert edge.edge_type == EdgeType.CONDITIONAL
        assert edge.target == {"continue": "a", "done": "b"}

    def test_set_entry_point(self):
        """set_entry_point should set entry."""
        graph = StateGraph(SimpleState)
        graph.add_node("start", increment_node)

        graph.set_entry_point("start")

        assert graph._entry_point == "start"

    def test_set_entry_point_invalid(self):
        """set_entry_point should raise for unknown node."""
        graph = StateGraph(SimpleState)

        with pytest.raises(ValueError, match="not found"):
            graph.set_entry_point("unknown")

    def test_set_finish_point(self):
        """set_finish_point should add edge to END."""
        graph = StateGraph(SimpleState)
        graph.add_node("last", increment_node)

        graph.set_finish_point("last")

        assert graph._edges["last"][0].target == END


class TestStateGraphCompile:
    """Tests for StateGraph compilation."""

    def test_compile_returns_compiled_graph(self):
        """compile should return CompiledGraph."""
        graph = StateGraph(SimpleState)
        graph.add_node("a", increment_node)
        graph.set_entry_point("a")
        graph.add_edge("a", END)

        compiled = graph.compile()

        assert isinstance(compiled, CompiledGraph)

    def test_compile_validates_nodes(self):
        """compile should fail if no nodes."""
        graph = StateGraph(SimpleState)

        with pytest.raises(ValueError, match="no nodes"):
            graph.compile()

    def test_compile_validates_entry_point(self):
        """compile should fail if no entry point."""
        graph = StateGraph(SimpleState)
        graph.add_node("a", increment_node)

        with pytest.raises(ValueError, match="No entry point"):
            graph.compile()

    def test_compile_validates_edge_targets(self):
        """compile should fail for invalid edge targets."""
        graph = StateGraph(SimpleState)
        graph.add_node("a", increment_node)
        graph.set_entry_point("a")
        graph.add_edge("a", "nonexistent")  # Invalid target

        with pytest.raises(ValueError, match="not found"):
            graph.compile()

    def test_compile_warns_on_implicit_dead_end(self, caplog):
        """compile should warn when a reachable node has no outgoing edges."""
        graph = StateGraph(SimpleState)
        graph.add_node("a", increment_node)
        graph.set_entry_point("a")

        with caplog.at_level(logging.WARNING):
            compiled = graph.compile()

        assert isinstance(compiled, CompiledGraph)
        assert "has no outgoing edges and will terminate implicitly" in caplog.text


# =============================================================================
# CompiledGraph Execution Tests
# =============================================================================


class TestCompiledGraphExecution:
    """Tests for CompiledGraph execution."""

    @pytest.fixture
    def simple_graph(self):
        """Create simple linear graph."""
        graph = StateGraph(SimpleState)
        graph.add_node("inc", increment_node)
        graph.add_node("double", double_node)
        graph.add_edge("inc", "double")
        graph.add_edge("double", END)
        graph.set_entry_point("inc")
        return graph.compile()

    @pytest.mark.asyncio
    async def test_invoke_linear_graph(self, simple_graph):
        """invoke should execute nodes in order."""
        state: SimpleState = {"value": 5, "history": []}

        result = await simple_graph.invoke(state)

        assert result.success is True
        assert result.state["value"] == 12  # (5+1)*2
        assert result.state["history"] == ["increment", "double"]
        assert result.node_history == ["inc", "double"]
        # state_history is opt-in (co-design review item 11): a plain
        # invoke() records none, since each snapshot is a full deep copy.
        assert result.state_history == []

    @pytest.mark.asyncio
    async def test_invoke_records_state_history_when_enabled(self):
        """record_state_history=True populates the per-node snapshot list."""
        graph = StateGraph(SimpleState)
        graph.add_node("inc", increment_node)
        graph.add_node("double", double_node)
        graph.add_edge("inc", "double")
        graph.add_edge("double", END)
        graph.set_entry_point("inc")
        compiled = graph.compile(record_state_history=True)

        state: SimpleState = {"value": 5, "history": []}
        result = await compiled.invoke(state)

        assert [node_id for node_id, _ in result.state_history] == ["inc", "double"]
        assert result.state_history[0][1]["value"] == 6
        assert result.state_history[1][1]["value"] == 12

    @pytest.mark.asyncio
    async def test_invoke_default_skips_state_snapshot_deepcopy(self, simple_graph, monkeypatch):
        """Perf negative: with the default off, no per-node state snapshot is
        taken at all — not just an empty list from a skipped append."""
        from victor.framework.graph_execution import snapshot_state_for_result as original

        calls = []

        def _spy(state):
            calls.append(state)
            return original(state)

        monkeypatch.setattr("victor.framework.graph.snapshot_state_for_result", _spy)

        state: SimpleState = {"value": 5, "history": []}
        await simple_graph.invoke(state)

        assert calls == []

    @pytest.mark.asyncio
    async def test_invoke_tracks_iterations(self, simple_graph):
        """invoke should track iterations."""
        state: SimpleState = {"value": 0, "history": []}

        result = await simple_graph.invoke(state)

        assert result.iterations == 2

    @pytest.mark.asyncio
    async def test_invoke_measures_duration(self, simple_graph):
        """invoke should measure duration."""
        state: SimpleState = {"value": 0, "history": []}

        result = await simple_graph.invoke(state)

        assert result.duration > 0

    @pytest.mark.asyncio
    async def test_invoke_with_cycle(self):
        """invoke should handle cyclic graphs."""
        graph = StateGraph(TaskState)
        graph.add_node("process", task_process_node)
        graph.add_node("check", task_check_node)

        graph.add_edge("process", "check")
        graph.add_conditional_edge(
            "check",
            should_continue,
            {"continue": "process", "done": END},
        )
        graph.set_entry_point("process")
        compiled = graph.compile()

        state: TaskState = {"task": "test", "iteration": 0, "complete": False}
        result = await compiled.invoke(state)

        assert result.success is True
        assert result.state["complete"] is True
        assert result.state["iteration"] == 3

    @pytest.mark.asyncio
    async def test_invoke_preserves_returned_pydantic_state_with_copy_on_write(self):
        """invoke should preserve a node's returned Pydantic model state."""

        async def increment_model(state: PydanticCounterState) -> PydanticCounterState:
            return state.model_copy(update={"iteration": state.iteration + 1})

        graph = StateGraph(PydanticCounterState)
        graph.add_node("increment", increment_model)
        graph.add_edge("increment", END)
        graph.set_entry_point("increment")

        result = await graph.compile().invoke(PydanticCounterState(iteration=0))

        assert result.success is True
        assert isinstance(result.state, PydanticCounterState)
        assert result.state.iteration == 1

    @pytest.mark.asyncio
    async def test_invoke_respects_max_iterations(self):
        """invoke should stop at max_iterations."""

        async def infinite_node(state: SimpleState) -> SimpleState:
            state["value"] += 1
            return state

        graph = StateGraph(SimpleState)
        graph.add_node("loop", infinite_node)
        graph.add_edge("loop", "loop")  # Infinite cycle
        graph.set_entry_point("loop")
        compiled = graph.compile(max_iterations=5)

        state: SimpleState = {"value": 0, "history": []}
        result = await compiled.invoke(state)

        assert result.success is False
        assert "Max iterations" in result.error
        assert result.state["value"] == 5  # Ran 5 times

    @pytest.mark.asyncio
    async def test_invoke_with_checkpointer(self):
        """invoke should save checkpoints."""
        checkpointer = MemoryCheckpointer()

        graph = StateGraph(SimpleState)
        graph.add_node("a", increment_node)
        graph.add_node("b", double_node)
        graph.add_edge("a", "b")
        graph.add_edge("b", END)
        graph.set_entry_point("a")
        compiled = graph.compile(checkpointer=checkpointer)

        state: SimpleState = {"value": 5, "history": []}
        await compiled.invoke(state, thread_id="test_thread")

        checkpoints = await checkpointer.list("test_thread")
        assert len(checkpoints) == 2  # One per node

    @pytest.mark.asyncio
    async def test_invoke_binds_graph_checkpoint_id_to_execution_checkpoint_context(
        self,
    ):
        """Graph checkpoints should bind their ID into execution checkpoint context."""
        checkpointer = MemoryCheckpointer()

        async def node(state):
            state["value"] += 1
            return state

        graph = StateGraph(dict)
        graph.add_node("a", node)
        graph.add_edge("a", END)
        graph.set_entry_point("a")
        compiled = graph.compile(checkpointer=checkpointer)
        execution_checkpoint = ExecutionCheckpoint.create(session_id="session-1")

        result = await compiled.invoke(
            {
                "value": 1,
                "context": {
                    "execution_checkpoint": execution_checkpoint,
                },
            },
            thread_id="test_thread",
        )

        checkpoints = await checkpointer.list("test_thread")
        assert len(checkpoints) == 1
        graph_checkpoint_id = checkpoints[0].checkpoint_id
        assert (
            result.state["context"]["execution_checkpoint_metadata"]["graph_checkpoint_id"]
            == graph_checkpoint_id
        )
        assert (
            result.state["context"]["execution_checkpoint"]["graph_checkpoint_id"]
            == graph_checkpoint_id
        )
        assert (
            checkpoints[0].state["context"]["execution_checkpoint_metadata"]["graph_checkpoint_id"]
            == graph_checkpoint_id
        )

    @pytest.mark.asyncio
    async def test_invoke_checkpoint_metadata_includes_plan_execution_state(self):
        """Graph checkpoints should expose plan state without state payload scraping."""
        checkpointer = MemoryCheckpointer()
        plan_execution_state = {
            "plan_id": "plan-1",
            "execution_mode": "team_adapter",
            "success": True,
            "step_statuses": {"1": "completed"},
        }

        async def node(state):
            state["value"] += 1
            return state

        graph = StateGraph(dict)
        graph.add_node("a", node)
        graph.add_edge("a", END)
        graph.set_entry_point("a")
        compiled = graph.compile(checkpointer=checkpointer)

        await compiled.invoke(
            {
                "value": 1,
                "plan_execution_state": plan_execution_state,
            },
            thread_id="test_thread",
        )

        checkpoints = await checkpointer.list("test_thread")
        assert len(checkpoints) == 1
        assert checkpoints[0].metadata["plan_execution_state"] == plan_execution_state

    @pytest.mark.asyncio
    async def test_invoke_parallel_fanout_can_continue_at_join_node(self):
        """Fan-out branches should be able to merge and continue via Send.join_at."""

        async def split_node(state: SimpleState) -> SimpleState:
            state["history"].append("split")
            return state

        def fan_out(state: SimpleState) -> List[Send]:
            return [
                Send(
                    "inc_branch",
                    {"value": state["value"], "history": list(state["history"])},
                    join_at="merge",
                ),
                Send(
                    "double_branch",
                    {"value": state["value"], "history": list(state["history"])},
                    join_at="merge",
                ),
            ]

        async def inc_branch(state: SimpleState) -> SimpleState:
            state["value"] += 1
            state["history"].append("inc_branch")
            return state

        async def double_branch(state: SimpleState) -> SimpleState:
            state["value"] *= 2
            state["history"].append("double_branch")
            return state

        async def merge_node(state: SimpleState) -> SimpleState:
            state["history"].append("merge")
            return state

        def merge_states(base: dict, branches: List[dict]) -> dict:
            history = list(base["history"])
            history.extend(branch["history"][-1] for branch in branches)
            return {
                "value": sum(branch["value"] for branch in branches),
                "history": history,
            }

        graph = StateGraph(SimpleState)
        graph.add_node("split", split_node)
        graph.add_node("inc_branch", inc_branch)
        graph.add_node("double_branch", double_branch)
        graph.add_node("merge", merge_node)
        graph.add_conditional_edge(
            "split",
            fan_out,
            {
                "inc_branch": "inc_branch",
                "double_branch": "double_branch",
                "join": "merge",
            },
        )
        graph.add_edge("merge", END)
        graph.add_state_merger(merge_states)
        graph.set_entry_point("split")

        result = await graph.compile().invoke({"value": 3, "history": []})

        assert result.success is True
        assert result.state["value"] == 10
        assert result.state["history"] == [
            "split",
            "inc_branch",
            "double_branch",
            "merge",
        ]
        assert result.node_history == [
            "split",
            "send:inc_branch",
            "send:double_branch",
            "merge",
        ]


class TestCompiledGraphStream:
    """Tests for CompiledGraph streaming."""

    @pytest.mark.asyncio
    async def test_stream_yields_states(self):
        """stream should yield state after each node."""
        graph = StateGraph(SimpleState)
        graph.add_node("a", increment_node)
        graph.add_node("b", double_node)
        graph.add_edge("a", "b")
        graph.add_edge("b", END)
        graph.set_entry_point("a")
        compiled = graph.compile()

        state: SimpleState = {"value": 5, "history": []}
        results = []

        async for node_id, node_state in compiled.stream(state):
            results.append((node_id, node_state["value"]))

        assert results == [("a", 6), ("b", 12)]

    @pytest.mark.asyncio
    async def test_stream_binds_graph_checkpoint_id_to_execution_checkpoint_context(
        self,
    ):
        """Streamed graph states should include bound graph checkpoint metadata."""
        checkpointer = MemoryCheckpointer()

        async def node(state):
            state["value"] += 1
            return state

        graph = StateGraph(dict)
        graph.add_node("a", node)
        graph.add_edge("a", END)
        graph.set_entry_point("a")
        compiled = graph.compile(checkpointer=checkpointer)
        execution_checkpoint = ExecutionCheckpoint.create(session_id="session-1")

        events = []
        async for node_id, node_state in compiled.stream(
            {
                "value": 1,
                "context": {
                    "execution_checkpoint": execution_checkpoint,
                },
            },
            thread_id="stream_thread",
        ):
            events.append((node_id, node_state))

        checkpoints = await checkpointer.list("stream_thread")
        assert len(checkpoints) == 1
        graph_checkpoint_id = checkpoints[0].checkpoint_id
        assert events[0][0] == "a"
        assert (
            events[0][1]["context"]["execution_checkpoint_metadata"]["graph_checkpoint_id"]
            == graph_checkpoint_id
        )

    @pytest.mark.asyncio
    async def test_stream_supports_parallel_fan_out_with_join(self):
        """stream should reuse the shared runtime path for fan-out workflows."""

        async def split_node(state: SimpleState) -> SimpleState:
            state["history"].append("split")
            return state

        def fan_out(state: SimpleState) -> List[Send]:
            return [
                Send(
                    node="inc_branch",
                    state={"value": state["value"], "history": list(state["history"])},
                    join_at="merge",
                ),
                Send(
                    node="double_branch",
                    state={"value": state["value"], "history": list(state["history"])},
                    join_at="merge",
                ),
            ]

        async def inc_branch(state: SimpleState) -> SimpleState:
            state["value"] += 1
            state["history"].append("inc_branch")
            return state

        async def double_branch(state: SimpleState) -> SimpleState:
            state["value"] *= 2
            state["history"].append("double_branch")
            return state

        async def merge_node(state: SimpleState) -> SimpleState:
            state["history"].append("merge")
            return state

        def merge_states(base: dict, branches: List[dict]) -> dict:
            history = list(base["history"])
            history.extend(branch["history"][-1] for branch in branches if branch["history"])
            return {
                "value": sum(branch["value"] for branch in branches),
                "history": history,
            }

        graph = StateGraph(SimpleState)
        graph.add_node("split", split_node)
        graph.add_node("inc_branch", inc_branch)
        graph.add_node("double_branch", double_branch)
        graph.add_node("merge", merge_node)
        graph.add_conditional_edge(
            "split",
            fan_out,
            {
                "inc_branch": "inc_branch",
                "double_branch": "double_branch",
                "join": "merge",
            },
        )
        graph.add_edge("merge", END)
        graph.add_state_merger(merge_states)
        graph.set_entry_point("split")

        results = []
        async for node_id, node_state in graph.compile().stream({"value": 3, "history": []}):
            results.append((node_id, node_state["value"], list(node_state["history"])))

        assert results == [
            ("split", 3, ["split"]),
            ("merge", 10, ["split", "inc_branch", "double_branch", "merge"]),
        ]


class TestCompiledGraphSchema:
    """Tests for graph schema extraction."""

    def test_get_graph_schema(self):
        """get_graph_schema should return graph structure."""
        graph = StateGraph(SimpleState)
        graph.add_node("a", increment_node)
        graph.add_node("b", double_node)
        graph.add_edge("a", "b")
        graph.add_edge("b", END)
        graph.set_entry_point("a")
        compiled = graph.compile()

        schema = compiled.get_graph_schema()

        assert set(schema["nodes"]) == {"a", "b"}
        assert schema["entry_point"] == "a"
        assert "a" in schema["edges"]


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateGraph:
    """Tests for create_graph factory."""

    def test_create_graph_returns_state_graph(self):
        """create_graph should return StateGraph."""
        graph = create_graph(SimpleState)

        assert isinstance(graph, StateGraph)
        assert graph._state_schema == SimpleState


# =============================================================================
# GraphConfig Tests
# =============================================================================


class TestGraphConfig:
    """Tests for GraphConfig."""

    def test_defaults(self):
        """GraphConfig should have sensible defaults."""
        config = GraphConfig()

        # Test focused config structure (ISP compliance)
        assert config.execution.max_iterations == 25
        assert config.execution.timeout is None
        assert config.execution.recursion_limit == 100
        assert config.checkpoint.checkpointer is None
        assert config.interrupt.interrupt_before == []
        assert config.interrupt.interrupt_after == []

    def test_custom_values(self):
        """GraphConfig should accept custom values via focused configs (ISP compliant)."""
        checkpointer = MemoryCheckpointer()
        config = GraphConfig(
            execution=ExecutionConfig(max_iterations=50, timeout=300.0),
            checkpoint=CheckpointConfig(checkpointer=checkpointer),
            interrupt=InterruptConfig(interrupt_before=["review"]),
        )

        assert config.execution.max_iterations == 50
        assert config.execution.timeout == 300.0
        assert config.checkpoint.checkpointer == checkpointer
        assert config.interrupt.interrupt_before == ["review"]

    def test_from_legacy_migration(self):
        """GraphConfig.from_legacy() should migrate from legacy format."""
        checkpointer = MemoryCheckpointer()
        config = GraphConfig.from_legacy(
            max_iterations=50,
            timeout=300.0,
            checkpointer=checkpointer,
            interrupt_before=["review"],
        )

        assert config.execution.max_iterations == 50
        assert config.execution.timeout == 300.0
        assert config.checkpoint.checkpointer == checkpointer
        assert config.interrupt.interrupt_before == ["review"]


# =============================================================================
# Framework Export Tests
# =============================================================================


class TestFrameworkExports:
    """Tests for framework module exports."""

    def test_graph_exported_from_framework(self):
        """Graph types should be exported from victor.framework."""
        from victor.framework import (
            StateGraph,
            CompiledGraph,
            Node,
            Edge,
            EdgeType,
            GraphExecutionResult,
            GraphConfig,
            WorkflowCheckpoint,
            MemoryCheckpointer,
            RLCheckpointerAdapter,
            END,
            START,
            create_graph,
        )

        assert StateGraph is not None
        assert CompiledGraph is not None
        assert END == "__end__"
        assert START == "__start__"

    def test_edge_types(self):
        """EdgeType enum should have correct values."""
        from victor.framework import EdgeType

        assert EdgeType.NORMAL.value == "normal"
        assert EdgeType.CONDITIONAL.value == "conditional"

    def test_node_status(self):
        """FrameworkNodeStatus enum should have correct values."""
        from victor.framework import FrameworkNodeStatus

        assert FrameworkNodeStatus.PENDING.value == "pending"
        assert FrameworkNodeStatus.RUNNING.value == "running"
        assert FrameworkNodeStatus.COMPLETED.value == "completed"
        assert FrameworkNodeStatus.FAILED.value == "failed"
        assert FrameworkNodeStatus.SKIPPED.value == "skipped"


# =============================================================================
# Subgraph Nesting Tests (Item 4)
# =============================================================================


class TestSubgraphNesting:
    """Tests for SubgraphNode and add_subgraph."""

    async def test_subgraph_basic_execution(self):
        """A subgraph node should execute its inner compiled graph."""
        # Inner graph: increment value
        inner = StateGraph(SimpleState)
        inner.add_node("inc", increment_node)
        inner.add_edge("inc", END)
        inner.set_entry_point("inc")
        inner_compiled = inner.compile()

        # Outer graph: uses subgraph
        outer = StateGraph(SimpleState)
        outer.add_node("start", sync_node)
        outer.add_subgraph("inner_graph", inner_compiled)
        outer.add_edge("start", "inner_graph")
        outer.add_edge("inner_graph", END)
        outer.set_entry_point("start")

        app = outer.compile()
        result = await app.invoke({"value": 0, "history": []})
        assert result.success
        # sync_node adds 10, inner increment_node adds 1
        assert result.state["value"] == 11

    async def test_subgraph_with_input_mapper(self):
        """Input mapper transforms parent state before subgraph."""
        inner = StateGraph(SimpleState)
        inner.add_node("double", double_node)
        inner.add_edge("double", END)
        inner.set_entry_point("double")
        inner_compiled = inner.compile()

        outer = StateGraph(SimpleState)
        outer.add_subgraph(
            "sub",
            inner_compiled,
            input_mapper=lambda s: {**s, "value": s["value"] + 100},
        )
        outer.add_edge("sub", END)
        outer.set_entry_point("sub")

        app = outer.compile()
        result = await app.invoke({"value": 5, "history": []})
        assert result.success
        # input_mapper: 5 + 100 = 105, double: 105 * 2 = 210
        assert result.state["value"] == 210

    async def test_subgraph_with_output_mapper(self):
        """Output mapper transforms subgraph result back to parent."""
        inner = StateGraph(SimpleState)
        inner.add_node("inc", increment_node)
        inner.add_edge("inc", END)
        inner.set_entry_point("inc")
        inner_compiled = inner.compile()

        outer = StateGraph(SimpleState)
        outer.add_subgraph(
            "sub",
            inner_compiled,
            output_mapper=lambda s: {**s, "value": s["value"] * 10},
        )
        outer.add_edge("sub", END)
        outer.set_entry_point("sub")

        app = outer.compile()
        result = await app.invoke({"value": 3, "history": []})
        assert result.success
        # inner: 3 + 1 = 4, output_mapper: 4 * 10 = 40
        assert result.state["value"] == 40

    async def test_subgraph_preserves_state_keys(self):
        """Subgraph should not lose parent state keys."""
        inner = StateGraph(SimpleState)
        inner.add_node("inc", increment_node)
        inner.add_edge("inc", END)
        inner.set_entry_point("inc")
        inner_compiled = inner.compile()

        outer = StateGraph(SimpleState)
        outer.add_subgraph("sub", inner_compiled)
        outer.add_edge("sub", END)
        outer.set_entry_point("sub")

        app = outer.compile()
        result = await app.invoke({"value": 1, "history": ["start"]})
        assert result.success
        assert "history" in result.state
        assert "increment" in result.state["history"]

    async def test_subgraph_node_in_history(self):
        """The subgraph node should appear in execution history."""
        inner = StateGraph(SimpleState)
        inner.add_node("inc", increment_node)
        inner.add_edge("inc", END)
        inner.set_entry_point("inc")
        inner_compiled = inner.compile()

        outer = StateGraph(SimpleState)
        outer.add_subgraph("sub", inner_compiled)
        outer.add_edge("sub", END)
        outer.set_entry_point("sub")

        app = outer.compile()
        result = await app.invoke({"value": 0, "history": []})
        assert "sub" in result.node_history

    def test_add_subgraph_duplicate_raises(self):
        """Adding a subgraph with an existing node ID should raise."""
        inner = StateGraph(SimpleState)
        inner.add_node("inc", increment_node)
        inner.add_edge("inc", END)
        inner.set_entry_point("inc")
        inner_compiled = inner.compile()

        graph = StateGraph(SimpleState)
        graph.add_node("dup", sync_node)
        with pytest.raises(ValueError, match="already exists"):
            graph.add_subgraph("dup", inner_compiled)

    async def test_nested_subgraph(self):
        """A subgraph can contain another subgraph (2-level nesting)."""
        # Innermost: increment
        innermost = StateGraph(SimpleState)
        innermost.add_node("inc", increment_node)
        innermost.add_edge("inc", END)
        innermost.set_entry_point("inc")
        innermost_compiled = innermost.compile()

        # Middle: wraps innermost
        middle = StateGraph(SimpleState)
        middle.add_subgraph("inner", innermost_compiled)
        middle.add_edge("inner", END)
        middle.set_entry_point("inner")
        middle_compiled = middle.compile()

        # Outer
        outer = StateGraph(SimpleState)
        outer.add_subgraph("mid", middle_compiled)
        outer.add_edge("mid", END)
        outer.set_entry_point("mid")

        app = outer.compile()
        result = await app.invoke({"value": 0, "history": []})
        assert result.success
        assert result.state["value"] == 1

    async def test_subgraph_depth_guard_prevents_infinite_recursion(
        self,
    ):
        """Self-referencing subgraph hits depth limit."""
        from victor.framework.graph import _MAX_SUBGRAPH_DEPTH

        # Create a graph that references itself via a subgraph
        inner = StateGraph(SimpleState)
        inner.add_node("inc", increment_node)
        inner.add_edge("inc", END)
        inner.set_entry_point("inc")
        inner_compiled = inner.compile()

        # Stack 12 levels of nesting (exceeds _MAX_SUBGRAPH_DEPTH=10)
        graphs = [inner_compiled]
        for i in range(12):
            wrapper = StateGraph(SimpleState)
            wrapper.add_subgraph("sub", graphs[-1])
            wrapper.add_edge("sub", END)
            wrapper.set_entry_point("sub")
            graphs.append(wrapper.compile())

        result = await graphs[-1].invoke({"value": 0, "history": []})
        # Should fail due to depth limit, not hang
        assert not result.success
        assert "depth" in result.error.lower() or "recursion" in result.error.lower()

    async def test_subgraph_depth_counter_not_leaked_to_parent(
        self,
    ):
        """Depth counter cleaned from output state."""
        from victor.framework.graph import _SUBGRAPH_DEPTH_KEY

        inner = StateGraph(SimpleState)
        inner.add_node("inc", increment_node)
        inner.add_edge("inc", END)
        inner.set_entry_point("inc")

        outer = StateGraph(SimpleState)
        outer.add_subgraph("sub", inner.compile())
        outer.add_edge("sub", END)
        outer.set_entry_point("sub")

        result = await outer.compile().invoke({"value": 0, "history": []})
        assert result.success
        assert _SUBGRAPH_DEPTH_KEY not in result.state

    async def test_subgraph_failure_propagates_as_error(self):
        """Subgraph that fails should surface the error."""

        async def failing_node(state):
            raise ValueError("intentional failure")

        inner = StateGraph(SimpleState)
        inner.add_node("fail", failing_node)
        inner.add_edge("fail", END)
        inner.set_entry_point("fail")

        outer = StateGraph(SimpleState)
        outer.add_subgraph("sub", inner.compile())
        outer.add_edge("sub", END)
        outer.set_entry_point("sub")

        result = await outer.compile().invoke({"value": 0, "history": []})
        assert not result.success
        assert "intentional failure" in result.error

    async def test_subgraph_from_schema(self):
        """from_schema should handle 'subgraph' node type."""
        inner = StateGraph(SimpleState)
        inner.add_node("inc", increment_node)
        inner.add_edge("inc", END)
        inner.set_entry_point("inc")
        inner_compiled = inner.compile()

        schema = {
            "nodes": [
                {"id": "start", "type": "function", "func": "sync_fn"},
                {"id": "sub", "type": "subgraph", "graph": "inner_graph"},
            ],
            "edges": [
                {"source": "start", "target": "sub", "type": "normal"},
                {"source": "sub", "target": END, "type": "normal"},
            ],
            "entry_point": "start",
        }

        graph = StateGraph.from_schema(
            schema,
            state_schema=SimpleState,
            node_registry={"sync_fn": sync_node, "inner_graph": inner_compiled},
        )
        app = graph.compile()
        result = await app.invoke({"value": 0, "history": []})
        assert result.success
        # sync_node adds 10, inner increment_node adds 1
        assert result.state["value"] == 11

    async def test_subgraph_node_dataclass(self):
        """SubgraphNode dataclass should store all attributes."""
        inner = StateGraph(SimpleState)
        inner.add_node("inc", increment_node)
        inner.add_edge("inc", END)
        inner.set_entry_point("inc")
        inner_compiled = inner.compile()

        node = SubgraphNode(id="test", compiled_graph=inner_compiled)
        assert node.id == "test"
        assert node.compiled_graph is inner_compiled
        assert node.input_mapper is None
        assert node.output_mapper is None

    async def test_subgraph_node_execute_directly(self):
        """SubgraphNode.execute() should work when called directly."""
        inner = StateGraph(SimpleState)
        inner.add_node("inc", increment_node)
        inner.add_edge("inc", END)
        inner.set_entry_point("inc")
        inner_compiled = inner.compile()

        node = SubgraphNode(id="test", compiled_graph=inner_compiled)
        result = await node.execute({"value": 5, "history": []})
        assert result["value"] == 6


# =============================================================================
# Send / Fan-out Tests (Item 5)
# =============================================================================


class TestDefaultStateMerger:
    """Tests for the default_state_merger function."""

    def test_merges_sequential(self):
        """Branch states are merged by sequential dict.update."""
        base = {"a": 1, "b": 2}
        branches = [{"a": 10, "c": 3}, {"d": 4}]
        merged = default_state_merger(base, branches)
        assert merged == {"a": 10, "b": 2, "c": 3, "d": 4}

    def test_empty_branches(self):
        """Empty branch list returns base state."""
        base = {"x": 1}
        merged = default_state_merger(base, [])
        assert merged == {"x": 1}

    def test_does_not_mutate_base(self):
        """The original base dict should not be mutated."""
        base = {"a": 1}
        branches = [{"a": 99}]
        default_state_merger(base, branches)
        assert base["a"] == 1

    def test_warns_on_conflicting_parallel_writes(self, caplog):
        """Conflicting keys should emit a last-write-wins warning."""
        base = {"a": 1}
        branches = [{"a": 10}, {"a": 20}]

        with caplog.at_level(logging.WARNING):
            merged = default_state_merger(base, branches)

        assert merged["a"] == 20
        assert "Parallel state merge conflict" in caplog.text

    def test_strict_merger_raises_on_conflicting_parallel_writes(self):
        """strict_state_merger should fail fast on conflicting keys."""
        base = {"a": 1}
        branches = [{"a": 10}, {"a": 20}]

        with pytest.raises(ValueError, match="Parallel state merge conflict"):
            strict_state_merger(base, branches)

    def test_strict_merger_merges_when_branches_do_not_conflict(self):
        """strict_state_merger should merge non-conflicting branches."""
        base = {"a": 1}
        branches = [{"b": 10}, {"c": 20}]

        merged = strict_state_merger(base, branches)

        assert merged == {"a": 1, "b": 10, "c": 20}


class TestSendFanOut:
    """Tests for Send dataclass and fan-out execution."""

    def test_send_dataclass(self):
        """Send stores node and state."""
        s = Send(node="process", state={"data": "hello"})
        assert s.node == "process"
        assert s.state == {"data": "hello"}

    def test_edge_returns_send_list(self):
        """A conditional edge returning List[Send] works."""

        def fanout_condition(state):
            return [
                Send(node="a", state={"val": 1}),
                Send(node="b", state={"val": 2}),
            ]

        edge = Edge(
            source="start",
            target={},
            edge_type=EdgeType.CONDITIONAL,
            condition=fanout_condition,
        )
        result = edge.get_target({"any": "state"})
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].node == "a"
        assert result[1].node == "b"

    def test_edge_returns_string_for_regular_condition(self):
        """A regular conditional edge still returns a string target."""

        def regular_condition(state):
            return "branch_a"

        edge = Edge(
            source="start",
            target={"branch_a": "node_a", "branch_b": "node_b"},
            edge_type=EdgeType.CONDITIONAL,
            condition=regular_condition,
        )
        result = edge.get_target({"any": "state"})
        assert result == "node_a"

    async def test_fan_out_parallel_execution(self):
        """Send-based fan-out executes branches in parallel and merges."""

        async def router_node(state):
            state["routed"] = True
            return state

        async def worker_a(state):
            state["a_result"] = state.get("a_val", 0) * 10
            return state

        async def worker_b(state):
            state["b_result"] = state.get("b_val", 0) + 100
            return state

        def fanout(state):
            return [
                Send(node="worker_a", state={**state, "a_val": 2}),
                Send(node="worker_b", state={**state, "b_val": 3}),
            ]

        graph = StateGraph()
        graph.add_node("router", router_node)
        graph.add_node("worker_a", worker_a)
        graph.add_node("worker_b", worker_b)
        graph.set_entry_point("router")
        # Map targets so validator sees them as reachable
        graph.add_conditional_edge("router", fanout, {"a": "worker_a", "b": "worker_b"})

        app = graph.compile()
        result = await app.invoke({"val": 0})
        assert result.success
        # Both branches should have executed and merged
        assert "a_result" in result.state
        assert "b_result" in result.state

    async def test_fan_out_with_custom_merger(self):
        """Custom state merger is used when set."""

        async def identity(state):
            return state

        def sum_merger(base, branches):
            merged = dict(base)
            for b in branches:
                for k, v in b.items():
                    if k in merged and isinstance(v, (int, float)):
                        merged[k] = merged.get(k, 0) + v
                    else:
                        merged[k] = v
            return merged

        def fanout(state):
            return [
                Send(node="a", state={"count": 10}),
                Send(node="b", state={"count": 20}),
            ]

        graph = StateGraph()
        graph.add_node("start", identity)
        graph.add_node("a", identity)
        graph.add_node("b", identity)
        graph.set_entry_point("start")
        graph.add_conditional_edge("start", fanout, {"x": "a", "y": "b"})
        graph.add_state_merger(sum_merger)

        app = graph.compile()
        result = await app.invoke({"count": 0})
        assert result.success
        # sum_merger: 0 + 10 + 20 = 30
        assert result.state["count"] == 30

    async def test_fan_out_with_strict_merger_fails_on_conflict(self):
        """strict_state_merger should surface conflicting branch writes."""

        async def identity(state):
            return state

        def fanout(state):
            return [
                Send(node="a", state={"count": 10}),
                Send(node="b", state={"count": 20}),
            ]

        graph = StateGraph()
        graph.add_node("start", identity)
        graph.add_node("a", identity)
        graph.add_node("b", identity)
        graph.set_entry_point("start")
        graph.add_conditional_edge("start", fanout, {"x": "a", "y": "b"})
        graph.add_state_merger(strict_state_merger)

        app = graph.compile()
        result = await app.invoke({"count": 0})
        assert result.success is False
        assert "Parallel state merge conflict" in result.error

    async def test_fan_out_defaults_to_strict_merge_strategy(self):
        """Parallel fan-out should fail fast on conflicting writes by default."""

        async def identity(state):
            return state

        def fanout(state):
            return [
                Send(node="a", state={"count": 10}),
                Send(node="b", state={"count": 20}),
            ]

        graph = StateGraph()
        graph.add_node("start", identity)
        graph.add_node("a", identity)
        graph.add_node("b", identity)
        graph.set_entry_point("start")
        graph.add_conditional_edge("start", fanout, {"x": "a", "y": "b"})

        result = await graph.compile().invoke({"count": 0})

        assert result.success is False
        assert "Parallel state merge conflict" in result.error

    async def test_fan_out_can_opt_into_last_write_wins_via_config(self):
        """Graphs can explicitly opt into legacy last-write-wins merge behavior."""

        async def identity(state):
            return state

        def fanout(state):
            return [
                Send(node="a", state={"count": 10}),
                Send(node="b", state={"count": 20}),
            ]

        graph = StateGraph()
        graph.add_node("start", identity)
        graph.add_node("a", identity)
        graph.add_node("b", identity)
        graph.set_entry_point("start")
        graph.add_conditional_edge("start", fanout, {"x": "a", "y": "b"})

        result = await graph.compile(
            parallel_state_merge_strategy="last_write_wins",
        ).invoke({"count": 0})

        assert result.success is True
        assert result.state["count"] == 20

    async def test_fan_out_node_history(self):
        """Fan-out branches appear in node_history as send:<node>."""

        async def noop(state):
            return state

        def fanout(state):
            return [
                Send(node="a", state=state),
                Send(node="b", state=state),
            ]

        graph = StateGraph()
        graph.add_node("router", noop)
        graph.add_node("a", noop)
        graph.add_node("b", noop)
        graph.set_entry_point("router")
        graph.add_conditional_edge("router", fanout, {"x": "a", "y": "b"})

        app = graph.compile()
        result = await app.invoke({})
        assert "send:a" in result.node_history
        assert "send:b" in result.node_history

    async def test_fan_out_fails_when_any_branch_errors(self):
        """Any failed branch should fail the fan-out with aggregated context."""

        async def identity(state):
            return state

        async def fail_branch(state):
            raise ParallelBranchExecutionError("worker_a branch exploded")

        def fanout(state):
            return [
                Send(node="a", state=state),
                Send(node="b", state=state),
            ]

        graph = StateGraph()
        graph.add_node("start", identity)
        graph.add_node("a", fail_branch)
        graph.add_node("b", identity)
        graph.set_entry_point("start")
        graph.add_conditional_edge("start", fanout, {"x": "a", "y": "b"})

        app = graph.compile()
        result = await app.invoke({})

        assert result.success is False
        assert "Parallel branch failure" in result.error
        assert "a:" in result.error

    async def test_fan_out_ends_graph_after_merge(self):
        """After fan-out, the graph should terminate (reach END)."""

        async def noop(state):
            return state

        def fanout(state):
            return [Send(node="a", state=state)]

        graph = StateGraph()
        graph.add_node("start", noop)
        graph.add_node("a", noop)
        graph.set_entry_point("start")
        graph.add_conditional_edge("start", fanout, {"x": "a"})

        app = graph.compile()
        result = await app.invoke({"x": 1})
        assert result.success

    def test_send_exported(self):
        """Send and default_state_merger are in __all__."""
        from victor.framework.graph import __all__ as graph_all

        assert "Send" in graph_all
        assert "default_state_merger" in graph_all
        assert "strict_state_merger" in graph_all
        assert "ParallelBranchExecutionError" in graph_all
        assert "SubgraphNode" in graph_all


# =============================================================================
# State History & Replay Tests (Item 6)
# =============================================================================


class TestStateHistoryReplay:
    """Tests for get_state_history, replay_from, and start_node."""

    async def test_get_state_history_returns_checkpoints(self):
        """get_state_history returns all checkpoints for a thread."""
        graph = StateGraph(SimpleState)
        graph.add_node("inc", increment_node)
        graph.add_node("double", double_node)
        graph.add_edge("inc", "double")
        graph.add_edge("double", END)
        graph.set_entry_point("inc")

        checkpointer = MemoryCheckpointer()
        app = graph.compile(checkpointer=checkpointer)

        thread_id = "test-history-thread"
        await app.invoke({"value": 0, "history": []}, thread_id=thread_id)

        history = await app.get_state_history(thread_id)
        assert len(history) >= 2  # at least one per node
        assert all(isinstance(cp, WorkflowCheckpoint) for cp in history)

    async def test_get_state_history_empty_without_checkpointer(self):
        """get_state_history returns [] when no checkpointer is configured."""
        graph = StateGraph(SimpleState)
        graph.add_node("inc", increment_node)
        graph.add_edge("inc", END)
        graph.set_entry_point("inc")

        app = graph.compile()
        history = await app.get_state_history("any-thread")
        assert history == []

    async def test_get_state_history_empty_for_unknown_thread(self):
        """get_state_history returns [] for a thread with no checkpoints."""
        graph = StateGraph(SimpleState)
        graph.add_node("inc", increment_node)
        graph.add_edge("inc", END)
        graph.set_entry_point("inc")

        checkpointer = MemoryCheckpointer()
        app = graph.compile(checkpointer=checkpointer)

        history = await app.get_state_history("nonexistent-thread")
        assert history == []

    async def test_start_node_skips_earlier_nodes(self):
        """invoke(start_node=...) starts execution from the given node."""
        graph = StateGraph(SimpleState)
        graph.add_node("inc", increment_node)
        graph.add_node("double", double_node)
        graph.add_edge("inc", "double")
        graph.add_edge("double", END)
        graph.set_entry_point("inc")

        app = graph.compile()

        # Skip "inc", start from "double" directly
        result = await app.invoke(
            {"value": 5, "history": []},
            start_node="double",
        )
        assert result.success
        # Only double (5 * 2 = 10), not increment
        assert result.state["value"] == 10
        assert "double" in result.node_history
        assert "inc" not in result.node_history

    async def test_replay_from_checkpoint(self):
        """replay_from restores checkpoint state and continues."""
        graph = StateGraph(SimpleState)
        graph.add_node("inc", increment_node)
        graph.add_node("double", double_node)
        graph.add_edge("inc", "double")
        graph.add_edge("double", END)
        graph.set_entry_point("inc")

        checkpointer = MemoryCheckpointer()
        app = graph.compile(checkpointer=checkpointer)

        thread_id = "replay-test"
        await app.invoke({"value": 0, "history": []}, thread_id=thread_id)

        history = await app.get_state_history(thread_id)
        assert len(history) > 0

        # Replay from first checkpoint (should be after "inc")
        first_cp = history[0]
        replay_result = await app.replay_from(thread_id, first_cp.checkpoint_id)
        assert replay_result.success

    async def test_replay_from_generates_new_thread(self):
        """replay_from creates a new thread to avoid polluting history."""
        graph = StateGraph(SimpleState)
        graph.add_node("inc", increment_node)
        graph.add_edge("inc", END)
        graph.set_entry_point("inc")

        checkpointer = MemoryCheckpointer()
        app = graph.compile(checkpointer=checkpointer)

        thread_id = "replay-thread-test"
        await app.invoke({"value": 0, "history": []}, thread_id=thread_id)

        history = await app.get_state_history(thread_id)
        first_cp = history[0]

        await app.replay_from(thread_id, first_cp.checkpoint_id)

        # Original thread history should be unchanged
        original_history = await app.get_state_history(thread_id)
        assert len(original_history) == len(history)

    async def test_replay_from_nonexistent_checkpoint_raises(self):
        """replay_from raises ValueError for unknown checkpoint_id."""
        graph = StateGraph(SimpleState)
        graph.add_node("inc", increment_node)
        graph.add_edge("inc", END)
        graph.set_entry_point("inc")

        checkpointer = MemoryCheckpointer()
        app = graph.compile(checkpointer=checkpointer)

        with pytest.raises(ValueError, match="not found"):
            await app.replay_from("any-thread", "nonexistent-checkpoint")

    async def test_replay_from_without_checkpointer_raises(self):
        """replay_from raises ValueError when no checkpointer."""
        graph = StateGraph(SimpleState)
        graph.add_node("inc", increment_node)
        graph.add_edge("inc", END)
        graph.set_entry_point("inc")

        app = graph.compile()

        with pytest.raises(ValueError, match="checkpointer"):
            await app.replay_from("any-thread", "any-checkpoint")

    async def test_state_history_ordered_chronologically(self):
        """Checkpoints should be ordered by creation time."""
        graph = StateGraph(SimpleState)
        graph.add_node("inc", increment_node)
        graph.add_node("double", double_node)
        graph.add_edge("inc", "double")
        graph.add_edge("double", END)
        graph.set_entry_point("inc")

        checkpointer = MemoryCheckpointer()
        app = graph.compile(checkpointer=checkpointer)

        thread_id = "ordered-test"
        await app.invoke({"value": 0, "history": []}, thread_id=thread_id)

        history = await app.get_state_history(thread_id)
        timestamps = [cp.timestamp for cp in history]
        assert timestamps == sorted(timestamps)

    async def test_replay_from_second_checkpoint(self):
        """Replaying from the second checkpoint uses its state."""
        graph = StateGraph(SimpleState)
        graph.add_node("inc", increment_node)
        graph.add_node("double", double_node)
        graph.add_edge("inc", "double")
        graph.add_edge("double", END)
        graph.set_entry_point("inc")

        checkpointer = MemoryCheckpointer()
        app = graph.compile(checkpointer=checkpointer)

        thread_id = "second-cp-test"
        await app.invoke({"value": 1, "history": []}, thread_id=thread_id)

        history = await app.get_state_history(thread_id)
        assert len(history) >= 2

        # Second checkpoint should be after "double" node
        second_cp = history[1]
        replay = await app.replay_from(thread_id, second_cp.checkpoint_id)
        assert replay.success
