"""Tests for BasicWorkflowGraph implementation.

These tests verify the graph structure, edge traversal, validation,
and cycle detection capabilities.
"""

import pytest
from typing import Any, Dict, Optional

import victor.workflows.graph as workflow_graph_module
from victor.workflows.protocols import (
    ProtocolNodeStatus,
    RetryPolicy,
    NodeResult,
    IWorkflowNode,
    IWorkflowEdge,
)
from victor.workflows.graph import (
    WorkflowNode,
    WorkflowEdge,
    ConditionalEdge,
    BasicWorkflowGraph,
    LegacyWorkflowGraph,
    DuplicateNodeError,
    InvalidEdgeError,
)

# Test fixtures


async def simple_handler(
    state: Dict[str, Any], context: Optional[Dict[str, Any]] = None
) -> NodeResult:
    """A simple handler that passes through state."""
    return NodeResult(status=ProtocolNodeStatus.COMPLETED, output=state)


async def increment_handler(
    state: Dict[str, Any], context: Optional[Dict[str, Any]] = None
) -> NodeResult:
    """Handler that increments a counter in state."""
    new_state = state.copy()
    new_state["counter"] = state.get("counter", 0) + 1
    return NodeResult(status=ProtocolNodeStatus.COMPLETED, output=new_state)


async def failing_handler(
    state: Dict[str, Any], context: Optional[Dict[str, Any]] = None
) -> NodeResult:
    """Handler that always fails."""
    return NodeResult(
        status=ProtocolNodeStatus.FAILED,
        error=ValueError("Intentional failure"),
    )


class TestWorkflowNode:
    """Tests for WorkflowNode class."""

    def test_node_creation(self):
        """WorkflowNode should be created with required fields."""
        node = WorkflowNode(
            id="test_node",
            name="Test Node",
            handler=simple_handler,
        )
        assert node.id == "test_node"
        assert node.name == "Test Node"
        assert node.handler == simple_handler

    def test_node_default_retry_policy(self):
        """WorkflowNode should have default retry policy."""
        node = WorkflowNode(
            id="test_node",
            name="Test Node",
            handler=simple_handler,
        )
        assert node.retry_policy.max_retries == 3
        assert node.retry_policy.delay_seconds == 1.0

    def test_node_custom_retry_policy(self):
        """WorkflowNode should accept custom retry policy."""
        custom_policy = RetryPolicy(max_retries=5, delay_seconds=2.0)
        node = WorkflowNode(
            id="test_node",
            name="Test Node",
            handler=simple_handler,
            retry_policy=custom_policy,
        )
        assert node.retry_policy.max_retries == 5
        assert node.retry_policy.delay_seconds == 2.0

    @pytest.mark.asyncio
    async def test_node_execute(self):
        """WorkflowNode execute should call handler."""
        node = WorkflowNode(
            id="test_node",
            name="Test Node",
            handler=increment_handler,
        )
        result = await node.execute({"counter": 5})
        assert result.status == ProtocolNodeStatus.COMPLETED
        assert result.output["counter"] == 6

    def test_node_implements_protocol(self):
        """WorkflowNode should implement IWorkflowNode protocol."""
        node = WorkflowNode(
            id="test_node",
            name="Test Node",
            handler=simple_handler,
        )
        assert isinstance(node, IWorkflowNode)


class TestWorkflowEdge:
    """Tests for WorkflowEdge class."""

    def test_edge_creation(self):
        """WorkflowEdge should be created with source and target."""
        edge = WorkflowEdge(source_id="node_a", target_id="node_b")
        assert edge.source_id == "node_a"
        assert edge.target_id == "node_b"

    def test_edge_always_traversable(self):
        """Simple WorkflowEdge should always be traversable."""
        edge = WorkflowEdge(source_id="node_a", target_id="node_b")
        assert edge.should_traverse({}) is True
        assert edge.should_traverse({"any": "state"}) is True

    def test_edge_implements_protocol(self):
        """WorkflowEdge should implement IWorkflowEdge protocol."""
        edge = WorkflowEdge(source_id="node_a", target_id="node_b")
        assert isinstance(edge, IWorkflowEdge)


class TestConditionalEdge:
    """Tests for ConditionalEdge class."""

    def test_conditional_edge_creation(self):
        """ConditionalEdge should be created with condition function."""

        def condition(state: Dict[str, Any]) -> bool:
            return state.get("proceed", False)

        edge = ConditionalEdge(
            source_id="check_node",
            target_id="next_node",
            condition=condition,
        )
        assert edge.source_id == "check_node"
        assert edge.target_id == "next_node"

    def test_conditional_edge_traversal(self):
        """ConditionalEdge should respect condition function."""

        def condition(state: Dict[str, Any]) -> bool:
            return state.get("success", False)

        edge = ConditionalEdge(
            source_id="check_node",
            target_id="success_node",
            condition=condition,
        )
        assert edge.should_traverse({"success": True}) is True
        assert edge.should_traverse({"success": False}) is False
        assert edge.should_traverse({}) is False

    def test_conditional_edge_with_state_value(self):
        """ConditionalEdge should work with complex state checks."""

        def condition(state: Dict[str, Any]) -> bool:
            return state.get("counter", 0) > 5

        edge = ConditionalEdge(
            source_id="loop_node",
            target_id="exit_node",
            condition=condition,
        )
        assert edge.should_traverse({"counter": 10}) is True
        assert edge.should_traverse({"counter": 3}) is False


class TestBasicWorkflowGraph:
    """Tests for BasicWorkflowGraph class."""

    def test_add_node_stores_node(self):
        """add_node should store the node in the graph."""
        graph = BasicWorkflowGraph()
        node = WorkflowNode(id="test_node", name="Test", handler=simple_handler)

        graph.add_node(node)

        assert graph.get_node("test_node") is node

    def test_add_node_fluent_interface(self):
        """add_node should return self for chaining."""
        graph = BasicWorkflowGraph()
        node1 = WorkflowNode(id="node1", name="Node 1", handler=simple_handler)
        node2 = WorkflowNode(id="node2", name="Node 2", handler=simple_handler)

        result = graph.add_node(node1).add_node(node2)

        assert result is graph
        assert graph.get_node("node1") is node1
        assert graph.get_node("node2") is node2

    def test_add_duplicate_node_raises_error(self):
        """add_node with duplicate ID should raise DuplicateNodeError."""
        graph = BasicWorkflowGraph()
        node1 = WorkflowNode(id="test_node", name="Test 1", handler=simple_handler)
        node2 = WorkflowNode(id="test_node", name="Test 2", handler=simple_handler)

        graph.add_node(node1)

        with pytest.raises(DuplicateNodeError) as exc_info:
            graph.add_node(node2)

        assert "test_node" in str(exc_info.value)

    def test_add_edge_validates_source_exists(self):
        """add_edge should validate source node exists."""
        graph = BasicWorkflowGraph()
        node = WorkflowNode(id="target_node", name="Target", handler=simple_handler)
        graph.add_node(node)

        edge = WorkflowEdge(source_id="nonexistent", target_id="target_node")

        with pytest.raises(InvalidEdgeError) as exc_info:
            graph.add_edge(edge)

        assert "nonexistent" in str(exc_info.value)
        assert "source" in str(exc_info.value).lower()

    def test_add_edge_validates_target_exists(self):
        """add_edge should validate target node exists."""
        graph = BasicWorkflowGraph()
        node = WorkflowNode(id="source_node", name="Source", handler=simple_handler)
        graph.add_node(node)

        edge = WorkflowEdge(source_id="source_node", target_id="nonexistent")

        with pytest.raises(InvalidEdgeError) as exc_info:
            graph.add_edge(edge)

        assert "nonexistent" in str(exc_info.value)
        assert "target" in str(exc_info.value).lower()

    def test_add_edge_fluent_interface(self):
        """add_edge should return self for chaining."""
        graph = BasicWorkflowGraph()
        node1 = WorkflowNode(id="node1", name="Node 1", handler=simple_handler)
        node2 = WorkflowNode(id="node2", name="Node 2", handler=simple_handler)
        node3 = WorkflowNode(id="node3", name="Node 3", handler=simple_handler)

        graph.add_node(node1).add_node(node2).add_node(node3)

        result = graph.add_edge(WorkflowEdge(source_id="node1", target_id="node2")).add_edge(
            WorkflowEdge(source_id="node2", target_id="node3")
        )

        assert result is graph

    def test_get_next_nodes_follows_edges(self):
        """get_next_nodes should return nodes connected by edges."""
        graph = BasicWorkflowGraph()
        node1 = WorkflowNode(id="node1", name="Node 1", handler=simple_handler)
        node2 = WorkflowNode(id="node2", name="Node 2", handler=simple_handler)
        node3 = WorkflowNode(id="node3", name="Node 3", handler=simple_handler)

        (
            graph.add_node(node1)
            .add_node(node2)
            .add_node(node3)
            .add_edge(WorkflowEdge(source_id="node1", target_id="node2"))
            .add_edge(WorkflowEdge(source_id="node1", target_id="node3"))
        )

        next_nodes = graph.get_next_nodes("node1", {})

        assert len(next_nodes) == 2
        assert node2 in next_nodes
        assert node3 in next_nodes

    def test_conditional_edge_evaluates_state(self):
        """get_next_nodes should respect conditional edge evaluation."""
        graph = BasicWorkflowGraph()
        node1 = WorkflowNode(id="check", name="Check", handler=simple_handler)
        node2 = WorkflowNode(id="success", name="Success", handler=simple_handler)
        node3 = WorkflowNode(id="failure", name="Failure", handler=simple_handler)

        def success_condition(state):
            return state.get("result") == "success"

        def failure_condition(state):
            return state.get("result") == "failure"

        (
            graph.add_node(node1)
            .add_node(node2)
            .add_node(node3)
            .add_edge(
                ConditionalEdge(source_id="check", target_id="success", condition=success_condition)
            )
            .add_edge(
                ConditionalEdge(source_id="check", target_id="failure", condition=failure_condition)
            )
        )

        # Test success path
        success_next = graph.get_next_nodes("check", {"result": "success"})
        assert len(success_next) == 1
        assert success_next[0].id == "success"

        # Test failure path
        failure_next = graph.get_next_nodes("check", {"result": "failure"})
        assert len(failure_next) == 1
        assert failure_next[0].id == "failure"

        # Test no match
        no_match = graph.get_next_nodes("check", {"result": "other"})
        assert len(no_match) == 0

    def test_set_entry_node(self):
        """set_entry_node should set the workflow entry point."""
        graph = BasicWorkflowGraph()
        node = WorkflowNode(id="start", name="Start", handler=simple_handler)
        graph.add_node(node)

        graph.set_entry_node("start")

        assert graph.get_entry_node() is node

    def test_set_entry_node_validates_exists(self):
        """set_entry_node should validate node exists."""
        graph = BasicWorkflowGraph()

        with pytest.raises(InvalidEdgeError):
            graph.set_entry_node("nonexistent")

    def test_set_exit_nodes(self):
        """set_exit_nodes should set terminal nodes."""
        graph = BasicWorkflowGraph()
        node1 = WorkflowNode(id="end1", name="End 1", handler=simple_handler)
        node2 = WorkflowNode(id="end2", name="End 2", handler=simple_handler)
        graph.add_node(node1).add_node(node2)

        graph.set_exit_nodes(["end1", "end2"])

        exit_nodes = graph.get_exit_nodes()
        assert len(exit_nodes) == 2
        assert node1 in exit_nodes
        assert node2 in exit_nodes

    def test_get_node_returns_none_for_missing(self):
        """get_node should return None for non-existent node."""
        graph = BasicWorkflowGraph()
        assert graph.get_node("nonexistent") is None


class TestGraphValidation:
    """Tests for graph validation."""

    def test_validate_empty_graph(self):
        """Empty graph should have validation errors."""
        graph = BasicWorkflowGraph()
        errors = graph.validate()

        assert len(errors) > 0
        assert any("entry" in e.lower() for e in errors)

    def test_validate_missing_entry_node(self):
        """Graph without entry node should have validation error."""
        graph = BasicWorkflowGraph()
        node = WorkflowNode(id="test", name="Test", handler=simple_handler)
        graph.add_node(node)

        errors = graph.validate()

        assert any("entry" in e.lower() for e in errors)

    def test_validate_detects_orphan_nodes(self):
        """validate should detect nodes not reachable from entry."""
        graph = BasicWorkflowGraph()
        node1 = WorkflowNode(id="start", name="Start", handler=simple_handler)
        node2 = WorkflowNode(id="connected", name="Connected", handler=simple_handler)
        node3 = WorkflowNode(id="orphan", name="Orphan", handler=simple_handler)

        (
            graph.add_node(node1)
            .add_node(node2)
            .add_node(node3)
            .add_edge(WorkflowEdge(source_id="start", target_id="connected"))
        )
        graph.set_entry_node("start")

        errors = graph.validate()

        assert any("orphan" in e.lower() for e in errors)

    def test_validate_valid_graph(self):
        """Valid graph should have no validation errors."""
        graph = BasicWorkflowGraph()
        node1 = WorkflowNode(id="start", name="Start", handler=simple_handler)
        node2 = WorkflowNode(id="process", name="Process", handler=simple_handler)
        node3 = WorkflowNode(id="end", name="End", handler=simple_handler)

        (
            graph.add_node(node1)
            .add_node(node2)
            .add_node(node3)
            .add_edge(WorkflowEdge(source_id="start", target_id="process"))
            .add_edge(WorkflowEdge(source_id="process", target_id="end"))
        )
        graph.set_entry_node("start")
        graph.set_exit_nodes(["end"])

        errors = graph.validate()

        assert len(errors) == 0


class TestCycleDetection:
    """Tests for cycle detection in graphs."""

    def test_detect_simple_cycle(self):
        """validate should detect simple A->B->A cycles."""
        graph = BasicWorkflowGraph()
        node1 = WorkflowNode(id="a", name="A", handler=simple_handler)
        node2 = WorkflowNode(id="b", name="B", handler=simple_handler)

        (
            graph.add_node(node1)
            .add_node(node2)
            .add_edge(WorkflowEdge(source_id="a", target_id="b"))
            .add_edge(WorkflowEdge(source_id="b", target_id="a"))
        )
        graph.set_entry_node("a")

        errors = graph.validate()

        assert any("cycle" in e.lower() for e in errors)

    def test_detect_self_loop(self):
        """validate should detect self-referencing nodes."""
        graph = BasicWorkflowGraph()
        node = WorkflowNode(id="loop", name="Loop", handler=simple_handler)
        node_end = WorkflowNode(id="end", name="End", handler=simple_handler)

        (
            graph.add_node(node)
            .add_node(node_end)
            .add_edge(WorkflowEdge(source_id="loop", target_id="loop"))
            .add_edge(WorkflowEdge(source_id="loop", target_id="end"))
        )
        graph.set_entry_node("loop")
        graph.set_exit_nodes(["end"])

        errors = graph.validate()

        assert any("cycle" in e.lower() for e in errors)

    def test_detect_complex_cycle(self):
        """validate should detect A->B->C->A cycles."""
        graph = BasicWorkflowGraph()
        node_a = WorkflowNode(id="a", name="A", handler=simple_handler)
        node_b = WorkflowNode(id="b", name="B", handler=simple_handler)
        node_c = WorkflowNode(id="c", name="C", handler=simple_handler)

        (
            graph.add_node(node_a)
            .add_node(node_b)
            .add_node(node_c)
            .add_edge(WorkflowEdge(source_id="a", target_id="b"))
            .add_edge(WorkflowEdge(source_id="b", target_id="c"))
            .add_edge(WorkflowEdge(source_id="c", target_id="a"))
        )
        graph.set_entry_node("a")

        errors = graph.validate()

        assert any("cycle" in e.lower() for e in errors)

    def test_no_false_positive_on_dag(self):
        """validate should not report cycles on valid DAGs."""
        graph = BasicWorkflowGraph()
        node_a = WorkflowNode(id="a", name="A", handler=simple_handler)
        node_b = WorkflowNode(id="b", name="B", handler=simple_handler)
        node_c = WorkflowNode(id="c", name="C", handler=simple_handler)
        node_d = WorkflowNode(id="d", name="D", handler=simple_handler)

        # Diamond pattern: A -> B, A -> C, B -> D, C -> D
        (
            graph.add_node(node_a)
            .add_node(node_b)
            .add_node(node_c)
            .add_node(node_d)
            .add_edge(WorkflowEdge(source_id="a", target_id="b"))
            .add_edge(WorkflowEdge(source_id="a", target_id="c"))
            .add_edge(WorkflowEdge(source_id="b", target_id="d"))
            .add_edge(WorkflowEdge(source_id="c", target_id="d"))
        )
        graph.set_entry_node("a")
        graph.set_exit_nodes(["d"])

        errors = graph.validate()

        # Should have no cycle errors
        assert not any("cycle" in e.lower() for e in errors)


class TestGraphHelperMethods:
    """Tests for graph helper methods."""

    def test_find_reachable_nodes(self):
        """_find_reachable_nodes should find all reachable nodes from entry."""
        graph = BasicWorkflowGraph()
        node_a = WorkflowNode(id="a", name="A", handler=simple_handler)
        node_b = WorkflowNode(id="b", name="B", handler=simple_handler)
        node_c = WorkflowNode(id="c", name="C", handler=simple_handler)
        node_orphan = WorkflowNode(id="orphan", name="Orphan", handler=simple_handler)

        (
            graph.add_node(node_a)
            .add_node(node_b)
            .add_node(node_c)
            .add_node(node_orphan)
            .add_edge(WorkflowEdge(source_id="a", target_id="b"))
            .add_edge(WorkflowEdge(source_id="b", target_id="c"))
        )
        graph.set_entry_node("a")

        reachable = graph._find_reachable_nodes()

        assert "a" in reachable
        assert "b" in reachable
        assert "c" in reachable
        assert "orphan" not in reachable

    def test_add_conditional_edge(self):
        """add_conditional_edge should create conditional branching."""
        graph = BasicWorkflowGraph()
        node_router = WorkflowNode(id="router", name="Router", handler=simple_handler)
        node_path_a = WorkflowNode(id="path_a", name="Path A", handler=simple_handler)
        node_path_b = WorkflowNode(id="path_b", name="Path B", handler=simple_handler)

        def router_func(state: Dict[str, Any]) -> str:
            return state.get("route", "path_a")

        (
            graph.add_node(node_router)
            .add_node(node_path_a)
            .add_node(node_path_b)
            .add_conditional_edge(
                source_id="router",
                router=router_func,
                targets={"path_a": "path_a", "path_b": "path_b"},
            )
        )

        # Test routing to path_a
        next_a = graph.get_next_nodes("router", {"route": "path_a"})
        assert len(next_a) == 1
        assert next_a[0].id == "path_a"

        # Test routing to path_b
        next_b = graph.get_next_nodes("router", {"route": "path_b"})
        assert len(next_b) == 1
        assert next_b[0].id == "path_b"


class TestWorkflowGraphCompatibilityAlias:
    """Tests for the deprecated WorkflowGraph compatibility alias."""

    def test_legacy_workflowgraph_alias_is_explicit(self):
        """LegacyWorkflowGraph should provide a non-warning explicit alias."""
        assert LegacyWorkflowGraph is BasicWorkflowGraph

    def test_workflowgraph_alias_warns_with_replacement_and_milestone(self):
        """WorkflowGraph alias should warn and return BasicWorkflowGraph."""
        with pytest.warns(
            DeprecationWarning,
            match=r"WorkflowGraph from victor\.workflows\.graph is deprecated",
        ) as caught:
            alias = workflow_graph_module.WorkflowGraph

        assert alias is BasicWorkflowGraph
        message = str(caught[0].message)
        assert "BasicWorkflowGraph" in message
        assert "victor.workflows.graph_dsl.WorkflowGraph" in message
        assert "v0.8.0" in message
        assert "2026-12-31" in message
        assert "docs/architecture/migration.md" in message
