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

Markers: integration
"""

from typing import TypedDict

import pytest

from victor.framework.graph import StateGraph, END

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
