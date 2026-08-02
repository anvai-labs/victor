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

"""Tests for StateGraph.from_schema() deserialization (Phase 3.0 prerequisite).

These tests verify the dynamic graph generation functionality that enables:
- Workflow persistence and restoration
- External graph definition (YAML, JSON)
- Dynamic workflow construction at runtime
"""

import pytest
from typing import TypedDict

from victor.framework.graph import StateGraph, END

# =============================================================================
# Test Fixtures
# =============================================================================


class SimpleState(TypedDict):
    """Simple state for testing."""

    value: int
    history: list[str]


class TaskState(TypedDict, total=False):
    """Task state with optional fields."""

    task: str
    result: str | None
    iteration: int
    complete: bool


# Node functions for testing
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


async def process_task_node(state: TaskState) -> TaskState:
    """Process task node."""
    state["iteration"] = state.get("iteration", 0) + 1
    return state


async def check_complete_node(state: TaskState) -> TaskState:
    """Check completion node."""
    if state.get("iteration", 0) >= 3:
        state["complete"] = True
        state["result"] = "Done"
    return state


def should_continue_condition(state: TaskState) -> str:
    """Condition function for branching."""
    if state.get("complete", False):
        return "done"
    return "continue"


# =============================================================================
# Test: Basic Deserialization
# =============================================================================


class TestBasicDeserialization:
    """Tests for basic from_schema() functionality."""

    def test_from_schema_dict_simple(self):
        """Should deserialize from simple dict schema."""
        schema = {
            "nodes": [
                {"id": "increment", "type": "function", "func": "increment"},
                {"id": "double", "type": "function", "func": "double"},
            ],
            "edges": [
                {"source": "increment", "target": "double", "type": "normal"},
                {"source": "double", "target": "__end__", "type": "normal"},
            ],
            "entry_point": "increment",
        }

        node_registry = {
            "increment": increment_node,
            "double": double_node,
        }

        graph = StateGraph.from_schema(schema, node_registry=node_registry)

        assert graph._entry_point == "increment"
        assert "increment" in graph._nodes
        assert "double" in graph._nodes
        assert len(graph._edges["increment"]) == 1
        assert graph._edges["increment"][0].target == "double"

    def test_from_schema_missing_required_fields(self):
        """Should raise ValueError if required fields are missing."""
        # Missing 'entry_point'
        schema = {
            "nodes": [{"id": "test", "type": "function", "func": "test"}],
            "edges": [],
        }

        with pytest.raises(ValueError, match="missing required fields"):
            StateGraph.from_schema(schema, node_registry={"test": increment_node})

    def test_from_schema_invalid_node_definition(self):
        """Should raise ValueError for invalid node definition."""
        schema = {
            "nodes": ["not_a_dict"],  # Invalid
            "edges": [],
            "entry_point": "test",
        }

        with pytest.raises(ValueError, match="Invalid node definition"):
            StateGraph.from_schema(schema, node_registry={})

    def test_from_schema_node_missing_id(self):
        """Should raise ValueError if node has no id."""
        schema = {
            "nodes": [{"type": "function"}],  # No 'id'
            "edges": [],
            "entry_point": "test",
        }

        with pytest.raises(ValueError, match="must have 'id' field"):
            StateGraph.from_schema(schema, node_registry={})

    def test_from_schema_function_node_missing_func(self):
        """Should raise ValueError if function node has no func."""
        schema = {
            "nodes": [{"id": "test", "type": "function"}],  # No 'func'
            "edges": [],
            "entry_point": "test",
        }

        with pytest.raises(ValueError, match="must specify 'func'"):
            StateGraph.from_schema(schema, node_registry={})

    def test_from_schema_function_not_in_registry(self):
        """Should raise ValueError if function not in registry."""
        schema = {
            "nodes": [{"id": "test", "type": "function", "func": "nonexistent"}],
            "edges": [],
            "entry_point": "test",
        }

        with pytest.raises(ValueError, match="not found in node_registry"):
            StateGraph.from_schema(schema, node_registry={})

    def test_from_schema_unsupported_node_type(self):
        """Should raise TypeError for unsupported node type."""
        schema = {
            "nodes": [{"id": "test", "type": "unknown_type"}],
            "edges": [],
            "entry_point": "test",
        }

        with pytest.raises(TypeError, match="Unsupported node type"):
            StateGraph.from_schema(schema, node_registry={})


# =============================================================================
# Test: Passthrough Nodes
# =============================================================================


class TestPassthroughNodes:
    """Tests for passthrough node type."""

    def test_passthrough_node(self):
        """Should create passthrough node that returns state unchanged."""
        schema = {
            "nodes": [
                {"id": "passthrough", "type": "passthrough"},
            ],
            "edges": [
                {"source": "passthrough", "target": "__end__", "type": "normal"},
            ],
            "entry_point": "passthrough",
        }

        graph = StateGraph.from_schema(schema)

        assert "passthrough" in graph._nodes
        # Passthrough node should be identity function
        # We can verify this by compilation
        compiled = graph.compile()
        assert compiled is not None


# =============================================================================
# Test: Edge Deserialization
# =============================================================================


class TestEdgeDeserialization:
    """Tests for edge deserialization."""

    def test_normal_edge(self):
        """Should deserialize normal edge."""
        schema = {
            "nodes": [
                {"id": "a", "type": "function", "func": "inc"},
                {"id": "b", "type": "function", "func": "double"},
            ],
            "edges": [
                {"source": "a", "target": "b", "type": "normal"},
            ],
            "entry_point": "a",
        }

        node_registry = {"inc": increment_node, "double": double_node}
        graph = StateGraph.from_schema(schema, node_registry=node_registry)

        assert "a" in graph._edges
        assert len(graph._edges["a"]) == 1
        assert graph._edges["a"][0].target == "b"

    def test_edge_to_end(self):
        """Should handle edges to END sentinel."""
        schema = {
            "nodes": [
                {"id": "a", "type": "function", "func": "inc"},
            ],
            "edges": [
                {"source": "a", "target": "__end__", "type": "normal"},
            ],
            "entry_point": "a",
        }

        graph = StateGraph.from_schema(schema, node_registry={"inc": increment_node})

        assert graph._edges["a"][0].target == END

    def test_edge_missing_source(self):
        """Should raise ValueError if edge has no source."""
        schema = {
            "nodes": [{"id": "test", "type": "function", "func": "test"}],
            "edges": [{"target": "__end__", "type": "normal"}],  # No 'source'
            "entry_point": "test",
        }

        with pytest.raises(ValueError, match="must have 'source' field"):
            StateGraph.from_schema(schema, node_registry={"test": increment_node})

    def test_edge_missing_target(self):
        """Should raise ValueError if edge has no target."""
        schema = {
            "nodes": [{"id": "test", "type": "function", "func": "test"}],
            "edges": [{"source": "test", "type": "normal"}],  # No 'target'
            "entry_point": "test",
        }

        with pytest.raises(ValueError, match="must have 'target' field"):
            StateGraph.from_schema(schema, node_registry={"test": increment_node})


# =============================================================================
# Test: Conditional Edges
# =============================================================================


class TestConditionalEdges:
    """Tests for conditional edge deserialization."""

    def test_conditional_edge(self):
        """Should deserialize conditional edge with condition function."""
        schema = {
            "nodes": [
                {"id": "process", "type": "function", "func": "process"},
                {"id": "check", "type": "function", "func": "check"},
            ],
            "edges": [
                {"source": "process", "target": "check", "type": "normal"},
                {
                    "source": "check",
                    "target": {"continue": "process", "done": "__end__"},
                    "type": "conditional",
                    "condition": "should_continue",
                },
            ],
            "entry_point": "process",
        }

        node_registry = {
            "process": process_task_node,
            "check": check_complete_node,
        }
        condition_registry = {
            "should_continue": should_continue_condition,
        }

        graph = StateGraph.from_schema(
            schema,
            node_registry=node_registry,
            condition_registry=condition_registry,
        )

        assert "check" in graph._edges
        edge = graph._edges["check"][0]
        # Verify it's a conditional edge
        from victor.framework.graph import EdgeType

        assert edge.edge_type == EdgeType.CONDITIONAL
        assert edge.target == {"continue": "process", "done": END}

    def test_conditional_edge_missing_condition(self):
        """Should raise ValueError if conditional edge missing condition."""
        schema = {
            "nodes": [
                {"id": "a", "type": "function", "func": "inc"},
            ],
            "edges": [
                {
                    "source": "a",
                    "target": {"b": "__end__"},
                    "type": "conditional",
                    # Missing 'condition'
                },
            ],
            "entry_point": "a",
        }

        with pytest.raises(ValueError, match="must specify 'condition'"):
            StateGraph.from_schema(schema, node_registry={"inc": increment_node})

    def test_conditional_edge_condition_not_in_registry(self):
        """Should raise ValueError if condition not in registry."""
        schema = {
            "nodes": [
                {"id": "a", "type": "function", "func": "inc"},
            ],
            "edges": [
                {
                    "source": "a",
                    "target": {"b": "__end__"},
                    "type": "conditional",
                    "condition": "nonexistent",
                },
            ],
            "entry_point": "a",
        }

        with pytest.raises(ValueError, match="not found in condition_registry"):
            StateGraph.from_schema(schema, node_registry={"inc": increment_node})

    def test_conditional_edge_invalid_target(self):
        """Should raise ValueError if conditional edge target is not dict."""
        schema = {
            "nodes": [
                {"id": "a", "type": "function", "func": "inc"},
            ],
            "edges": [
                {
                    "source": "a",
                    "target": "not_a_dict",  # Should be dict
                    "type": "conditional",
                    "condition": "should_continue",
                },
            ],
            "entry_point": "a",
        }

        with pytest.raises(ValueError, match="must be dict mapping branches"):
            StateGraph.from_schema(
                schema,
                node_registry={"inc": increment_node},
                condition_registry={"should_continue": should_continue_condition},
            )


# =============================================================================
# Test: Entry Point
# =============================================================================


class TestEntryPoint:
    """Tests for entry point validation."""

    def test_entry_point_not_found(self):
        """Should raise ValueError if entry point not in nodes."""
        schema = {
            "nodes": [
                {"id": "a", "type": "function", "func": "inc"},
            ],
            "edges": [],
            "entry_point": "nonexistent",  # Not in nodes
        }

        with pytest.raises(ValueError, match="Entry point.*not found in nodes"):
            StateGraph.from_schema(schema, node_registry={"inc": increment_node})


# =============================================================================
# Test: YAML Deserialization
# =============================================================================


class TestYAMLDeserialization:
    """Tests for YAML string deserialization."""

    def test_from_schema_yaml_simple(self):
        """Should deserialize from YAML string."""
        yaml_schema = """
nodes:
  - id: increment
    type: function
    func: increment
  - id: double
    type: function
    func: double
edges:
  - source: increment
    target: double
    type: normal
  - source: double
    target: __end__
    type: normal
entry_point: increment
"""

        node_registry = {
            "increment": increment_node,
            "double": double_node,
        }

        graph = StateGraph.from_schema(yaml_schema, node_registry=node_registry)

        assert graph._entry_point == "increment"
        assert "increment" in graph._nodes
        assert "double" in graph._nodes

    def test_from_schema_yaml_conditional(self):
        """Should deserialize conditional edges from YAML."""
        yaml_schema = """
nodes:
  - id: process
    type: function
    func: process
  - id: check
    type: function
    func: check
edges:
  - source: process
    target: check
    type: normal
  - source: check
    target:
      continue: process
      done: __end__
    type: conditional
    condition: should_continue
entry_point: process
"""

        node_registry = {
            "process": process_task_node,
            "check": check_complete_node,
        }
        condition_registry = {
            "should_continue": should_continue_condition,
        }

        graph = StateGraph.from_schema(
            yaml_schema,
            node_registry=node_registry,
            condition_registry=condition_registry,
        )

        assert graph._entry_point == "process"
        assert len(graph._edges["check"]) == 1
        from victor.framework.graph import EdgeType

        assert graph._edges["check"][0].edge_type == EdgeType.CONDITIONAL

    def test_from_schema_invalid_yaml(self):
        """Should raise ValueError for invalid YAML."""
        invalid_yaml = """
nodes:
  - this is: [invalid: yaml
    unclosed brackets
"""

        with pytest.raises(ValueError, match="Invalid YAML"):
            StateGraph.from_schema(invalid_yaml, node_registry={})


# =============================================================================
# Test: Metadata Support
# =============================================================================


class TestMetadataSupport:
    """Tests for metadata preservation in deserialization."""

    def test_node_metadata(self):
        """Should preserve node metadata from schema."""
        schema = {
            "nodes": [
                {
                    "id": "increment",
                    "type": "function",
                    "func": "inc",
                    "description": "Increments value",
                    "timeout": 30,
                },
            ],
            "edges": [
                {"source": "increment", "target": "__end__", "type": "normal"},
            ],
            "entry_point": "increment",
        }

        graph = StateGraph.from_schema(schema, node_registry={"inc": increment_node})

        node = graph._nodes["increment"]
        assert node.metadata.get("description") == "Increments value"
        assert node.metadata.get("timeout") == 30

    def test_graph_metadata(self):
        """Should preserve graph-level metadata from schema."""
        schema = {
            "nodes": [
                {"id": "increment", "type": "function", "func": "inc"},
            ],
            "edges": [
                {"source": "increment", "target": "__end__", "type": "normal"},
            ],
            "entry_point": "increment",
            "metadata": {"owner": "schema-loader", "version": 2},
        }

        graph = StateGraph.from_schema(schema, node_registry={"inc": increment_node})

        assert graph.metadata == {"owner": "schema-loader", "version": 2}


# =============================================================================
# Test: State Schema Support
# =============================================================================


class TestStateSchemaSupport:
    """Tests for state_schema parameter."""

    def test_state_schema_passed(self):
        """Should pass state_schema to StateGraph."""
        schema = {
            "nodes": [
                {"id": "test", "type": "function", "func": "inc"},
            ],
            "edges": [
                {"source": "test", "target": "__end__", "type": "normal"},
            ],
            "entry_point": "test",
        }

        graph = StateGraph.from_schema(
            schema, state_schema=SimpleState, node_registry={"inc": increment_node}
        )

        assert graph._state_schema == SimpleState


# =============================================================================
# Test: End-to-End Execution
# =============================================================================


class TestEndToEndExecution:
    """Tests for full execution of deserialized graphs."""

    @pytest.mark.asyncio
    async def test_execute_deserialized_graph(self):
        """Should execute graph deserialized from schema."""
        schema = {
            "nodes": [
                {"id": "increment", "type": "function", "func": "increment"},
                {"id": "double", "type": "function", "func": "double"},
            ],
            "edges": [
                {"source": "increment", "target": "double", "type": "normal"},
                {"source": "double", "target": "__end__", "type": "normal"},
            ],
            "entry_point": "increment",
        }

        node_registry = {
            "increment": increment_node,
            "double": double_node,
        }

        graph = StateGraph.from_schema(schema, node_registry=node_registry)
        compiled = graph.compile()

        initial_state: SimpleState = {"value": 5, "history": []}
        result = await compiled.invoke(initial_state)

        assert result.state["value"] == 12  # (5 + 1) * 2
        assert result.state["history"] == ["increment", "double"]
        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_conditional_graph(self):
        """Should execute conditional graph from schema."""
        schema = {
            "nodes": [
                {"id": "process", "type": "function", "func": "process_task"},
                {"id": "check", "type": "function", "func": "check_complete"},
            ],
            "edges": [
                {"source": "process", "target": "check", "type": "normal"},
                {
                    "source": "check",
                    "target": {"continue": "process", "done": "__end__"},
                    "type": "conditional",
                    "condition": "should_continue",
                },
            ],
            "entry_point": "process",
        }

        node_registry = {
            "process_task": process_task_node,
            "check_complete": check_complete_node,
        }
        condition_registry = {
            "should_continue": should_continue_condition,
        }

        graph = StateGraph.from_schema(
            schema,
            node_registry=node_registry,
            condition_registry=condition_registry,
        )
        compiled = graph.compile()

        initial_state: TaskState = {}
        result = await compiled.invoke(initial_state)

        # Should iterate 3 times before completion
        assert result.state["iteration"] == 3
        assert result.state["complete"] is True
        assert result.success is True


# =============================================================================
# Test: Round-Trip Serialization
# =============================================================================


class TestRoundTripSerialization:
    """Tests for serialization/deserialization round-trip."""

    @pytest.mark.asyncio
    async def test_round_trip_get_graph_schema(self):
        """Should be able to serialize and deserialize graph."""
        # Create original graph
        original_graph = StateGraph()
        original_graph.add_node("increment", increment_node)
        original_graph.add_node("double", double_node)
        original_graph.add_edge("increment", "double")
        original_graph.add_edge("double", END)
        original_graph.set_entry_point("increment")

        # Compile and get schema
        compiled = original_graph.compile()
        schema = compiled.get_graph_schema()

        # Create schema format for from_schema
        deserialization_schema = {
            "nodes": [
                {"id": node_id, "type": "function", "func": node_id} for node_id in schema["nodes"]
            ],
            "edges": [],
            "entry_point": schema["entry_point"],
        }

        # Convert edges to from_schema format
        for source, edge_list in schema["edges"].items():
            for edge_info in edge_list:
                deserialization_schema["edges"].append(
                    {
                        "source": source,
                        "target": edge_info["target"],
                        "type": edge_info["type"],
                    }
                )

        # Deserialize
        node_registry = {
            "increment": increment_node,
            "double": double_node,
        }

        restored_graph = StateGraph.from_schema(deserialization_schema, node_registry=node_registry)

        # Verify structure matches
        assert restored_graph._entry_point == original_graph._entry_point
        assert set(restored_graph._nodes.keys()) == set(original_graph._nodes.keys())

        # Execute both and verify same result
        initial_state: SimpleState = {"value": 5, "history": []}

        original_result = await compiled.invoke(initial_state)
        restored_result = await restored_graph.compile().invoke(initial_state)

        assert original_result.state == restored_result.state
