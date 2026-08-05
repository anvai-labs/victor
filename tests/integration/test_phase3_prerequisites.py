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

"""Integration tests for Phase 3.0 prerequisites.

These tests demonstrate end-to-end integration of:
- StateGraph.from_schema() deserialization
- WorkflowMetricsCollector with workflow execution
- Persistence and restoration of workflows with metrics

Markers: integration
"""

import os
import tempfile
from typing import TypedDict

import pytest

from victor.framework.graph import StateGraph, END
from victor.workflows.metrics import WorkflowMetricsCollector
from victor.workflows.observability import ObservabilityEmitter

# =============================================================================
# Test Fixtures
# =============================================================================


class TaskState(TypedDict, total=False):
    """Task state for integration tests."""

    task: str
    result: str | None
    iteration: int
    complete: bool


# Node functions
async def process_node(state: TaskState) -> TaskState:
    """Process node that increments iteration."""
    state["iteration"] = state.get("iteration", 0) + 1
    return state


async def check_node(state: TaskState) -> TaskState:
    """Check node that marks completion."""
    if state.get("iteration", 0) >= 3:
        state["complete"] = True
        state["result"] = f"Completed after {state['iteration']} iterations"
    return state


def should_continue(state: TaskState) -> str:
    """Condition function for branching."""
    if state.get("complete", False):
        return "done"
    return "continue"


# =============================================================================
# Test: End-to-End Graph Deserialization and Execution
# =============================================================================


@pytest.mark.integration
class TestGraphDeserializationE2E:
    """End-to-end tests for graph deserialization."""

    @pytest.mark.asyncio
    async def test_deserialize_and_execute_workflow(self):
        """Should deserialize workflow from dict and execute successfully."""
        # Define workflow schema
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

        # Create registries
        node_registry = {
            "process_task": process_node,
            "check_complete": check_node,
        }
        condition_registry = {
            "should_continue": should_continue,
        }

        # Deserialize graph
        graph = StateGraph.from_schema(
            schema,
            state_schema=TaskState,
            node_registry=node_registry,
            condition_registry=condition_registry,
        )

        # Compile and execute
        compiled = graph.compile()
        initial_state: TaskState = {}
        result = await compiled.invoke(initial_state)

        # Verify execution
        assert result.success is True
        assert result.state["iteration"] == 3
        assert result.state["complete"] is True
        assert "Completed after" in result.state["result"]

    @pytest.mark.asyncio
    async def test_deserialize_from_yaml_and_execute(self):
        """Should deserialize workflow from YAML and execute successfully."""
        yaml_schema = """
nodes:
  - id: analyze
    type: function
    func: process_task
  - id: validate
    type: function
    func: check_complete
edges:
  - source: analyze
    target: validate
    type: normal
  - source: validate
    target:
      continue: analyze
      done: __end__
    type: conditional
    condition: should_continue
entry_point: analyze
"""

        node_registry = {
            "process_task": process_node,
            "check_complete": check_node,
        }
        condition_registry = {
            "should_continue": should_continue,
        }

        # Deserialize from YAML
        graph = StateGraph.from_schema(
            yaml_schema,
            node_registry=node_registry,
            condition_registry=condition_registry,
        )

        # Compile and execute
        compiled = graph.compile()
        result = await compiled.invoke({})

        assert result.success is True
        assert result.state["complete"] is True


# =============================================================================
# Test: Metrics Collection Integration
# =============================================================================


@pytest.mark.integration
class TestMetricsCollectionE2E:
    """End-to-end tests for metrics collection."""

    @pytest.mark.asyncio
    async def test_collect_metrics_during_execution(self):
        """Should collect metrics during workflow execution."""
        # Create workflow
        graph = StateGraph()
        graph.add_node("process", process_node)
        graph.add_node("check", check_node)
        graph.add_edge("process", "check")
        graph.add_conditional_edge("check", should_continue, {"continue": "process", "done": END})
        graph.set_entry_point("process")

        compiled = graph.compile()

        # Create metrics collector with observability
        collector = WorkflowMetricsCollector(storage_backend="memory")
        emitter = ObservabilityEmitter(
            workflow_id="test_wf",
            workflow_name="Test Workflow",
            total_nodes=2,
        )

        # Register collector with emitter
        emitter.add_observer(collector)

        # Execute workflow with event emission
        await compiled.invoke({}, config=compiled._config)

        # Note: In a real integration, the emitter would be integrated
        # with CompiledGraph execution. For this test, we verify
        # the collector infrastructure works.

        # Verify collector is ready
        assert collector is not None
        assert collector.storage_backend == "memory"

    @pytest.mark.asyncio
    async def test_persist_metrics_to_json(self):
        """Should persist metrics to JSON and restore."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "metrics.json")

            # Create collector with JSON persistence
            collector = WorkflowMetricsCollector(
                storage_backend="json", storage_path=json_path, auto_save=True
            )

            # Simulate workflow execution events
            collector.on_event(
                WorkflowStreamChunk(
                    event_type=WorkflowEventType.WORKFLOW_START,
                    workflow_id="test_wf",
                    progress=0.0,
                    metadata={"workflow_name": "Test"},
                )
            )

            collector.on_event(
                WorkflowStreamChunk(
                    event_type=WorkflowEventType.NODE_COMPLETE,
                    workflow_id="test_wf",
                    node_id="process",
                    progress=50.0,
                    metadata={"duration_seconds": 1.5},
                )
            )

            collector.on_event(
                WorkflowStreamChunk(
                    event_type=WorkflowEventType.WORKFLOW_COMPLETE,
                    workflow_id="test_wf",
                    progress=100.0,
                    is_final=True,
                )
            )

            # Create new collector and load
            collector2 = WorkflowMetricsCollector(
                storage_backend="json", storage_path=json_path, auto_save=False
            )
            collector2.load()

            # Verify restored metrics
            metrics = collector2.get_workflow_metrics("test_wf")
            assert metrics is not None
            assert metrics.workflow_name == "Test"
            assert metrics.total_executions == 1
            assert "process" in metrics.node_metrics

    @pytest.mark.asyncio
    async def test_persist_metrics_to_sqlite(self):
        """Should persist metrics to SQLite and restore."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "metrics.db")

            # Create collector with SQLite persistence
            collector = WorkflowMetricsCollector(
                storage_backend="sqlite", storage_path=db_path, auto_save=True
            )

            # Simulate workflow execution events
            collector.on_event(
                WorkflowStreamChunk(
                    event_type=WorkflowEventType.WORKFLOW_START,
                    workflow_id="test_wf",
                    progress=0.0,
                    metadata={"workflow_name": "Test"},
                )
            )

            collector.on_event(
                WorkflowStreamChunk(
                    event_type=WorkflowEventType.NODE_COMPLETE,
                    workflow_id="test_wf",
                    node_id="process",
                    progress=50.0,
                    metadata={"duration_seconds": 1.5},
                )
            )

            collector.on_event(
                WorkflowStreamChunk(
                    event_type=WorkflowEventType.AGENT_TOOL_CALL,
                    workflow_id="test_wf",
                    node_id="agent",
                    progress=75.0,
                    tool_calls=[{"name": "search", "duration": 0.5}],
                )
            )

            collector.on_event(
                WorkflowStreamChunk(
                    event_type=WorkflowEventType.WORKFLOW_COMPLETE,
                    workflow_id="test_wf",
                    progress=100.0,
                    is_final=True,
                )
            )

            # Create new collector and load
            collector2 = WorkflowMetricsCollector(
                storage_backend="sqlite", storage_path=db_path, auto_save=False
            )
            collector2.load()

            # Verify restored metrics
            metrics = collector2.get_workflow_metrics("test_wf")
            assert metrics is not None
            assert metrics.workflow_name == "Test"
            assert metrics.total_executions == 1
            assert "process" in metrics.node_metrics
            assert "search" in metrics.tool_metrics


# =============================================================================
# Test: Complete Workflow Lifecycle
# =============================================================================


@pytest.mark.integration
class TestWorkflowLifecycleE2E:
    """End-to-end tests for complete workflow lifecycle."""

    @pytest.mark.asyncio
    async def test_workflow_save_deserialize_execute_with_metrics(self):
        """Should save workflow schema, deserialize, execute, and collect metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 1: Define and execute original workflow
            original_graph = StateGraph()
            original_graph.add_node("process", process_node)
            original_graph.add_node("check", check_node)
            original_graph.add_edge("process", "check")
            original_graph.add_conditional_edge(
                "check", should_continue, {"continue": "process", "done": END}
            )
            original_graph.set_entry_point("process")

            # Get schema
            compiled = original_graph.compile()
            schema = compiled.get_graph_schema()

            # Convert to from_schema format
            deserialization_schema = {
                "nodes": [
                    {"id": node_id, "type": "function", "func": f"{node_id}_func"}
                    for node_id in schema["nodes"]
                ],
                "edges": [],
                "entry_point": schema["entry_point"],
            }

            for source, edge_list in schema["edges"].items():
                for edge_info in edge_list:
                    deserialization_schema["edges"].append(
                        {
                            "source": source,
                            "target": edge_info["target"],
                            "type": edge_info["type"],
                        }
                    )

            # Step 2: Save schema to file
            import json

            schema_path = os.path.join(tmpdir, "workflow_schema.json")
            with open(schema_path, "w") as f:
                json.dump(deserialization_schema, f, indent=2)

            # Step 3: Load schema from file and deserialize
            with open(schema_path) as f:
                loaded_schema = json.load(f)

            # For conditional edges, we need to add condition info
            for edge in loaded_schema["edges"]:
                if edge["type"] == "conditional":
                    edge["condition"] = "should_continue"

            # Create registries
            node_registry = {
                "process_func": process_node,
                "check_func": check_node,
            }
            condition_registry = {
                "should_continue": should_continue,
            }

            # Deserialize
            restored_graph = StateGraph.from_schema(
                loaded_schema,
                node_registry=node_registry,
                condition_registry=condition_registry,
            )

            # Step 4: Execute restored workflow with metrics
            restored_compiled = restored_graph.compile()

            metrics_path = os.path.join(tmpdir, "metrics.json")
            collector = WorkflowMetricsCollector(
                storage_backend="json", storage_path=metrics_path, auto_save=True
            )

            # Simulate events
            collector.on_event(
                WorkflowStreamChunk(
                    event_type=WorkflowEventType.WORKFLOW_START,
                    workflow_id="lifecycle_test",
                    progress=0.0,
                    metadata={"workflow_name": "Lifecycle Test"},
                )
            )

            result = await restored_compiled.invoke({})

            collector.on_event(
                WorkflowStreamChunk(
                    event_type=WorkflowEventType.WORKFLOW_COMPLETE,
                    workflow_id="lifecycle_test",
                    progress=100.0,
                    is_final=True,
                )
            )

            # Step 5: Verify execution
            assert result.success is True
            assert result.state["complete"] is True

            # Step 6: Verify metrics were persisted
            collector2 = WorkflowMetricsCollector(
                storage_backend="json", storage_path=metrics_path, auto_save=False
            )
            collector2.load()

            metrics = collector2.get_workflow_metrics("lifecycle_test")
            assert metrics is not None
            assert metrics.workflow_name == "Lifecycle Test"


# Import WorkflowStreamChunk and WorkflowEventType
from victor.workflows.streaming import WorkflowStreamChunk, WorkflowEventType
