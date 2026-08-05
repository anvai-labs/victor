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

"""Integration tests for workflow state management.

These tests verify state handling across workflow execution:
- State persistence across nodes
- State modification propagation
- Large state handling
- State isolation between workflow instances
- State serialization and deserialization
"""

import asyncio
import copy
import json
import pytest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from victor.workflows.graph_dsl import WorkflowGraph, State
from victor.workflows.definition import WorkflowBuilder, TransformNode
from victor.workflows.context import WorkflowContext, WorkflowResult
from victor.workflows.unified_executor import WorkflowExecutor
from victor_contracts.workflows import ExecutorNodeStatus, NodeResult

# ============ Test State Classes ============


@dataclass
class PersistenceState(State):
    """State for persistence tests."""

    values: List[str] = field(default_factory=list)
    counter: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LargeDataState(State):
    """State for large data handling tests."""

    large_list: List[Dict[str, Any]] = field(default_factory=list)
    nested_data: Dict[str, Any] = field(default_factory=dict)
    binary_data: Optional[bytes] = None


@dataclass
class IsolationState(State):
    """State for isolation tests."""

    instance_id: str = ""
    modifications: List[str] = field(default_factory=list)


@dataclass
class ComplexState(State):
    """Complex nested state for advanced tests."""

    config: Dict[str, Any] = field(default_factory=dict)
    items: List[Dict[str, Any]] = field(default_factory=list)
    cache: Dict[str, List[str]] = field(default_factory=dict)
    counters: Dict[str, int] = field(default_factory=dict)


# ============ Helper Functions ============


def create_mock_orchestrator():
    """Create a mock orchestrator for testing."""
    mock = MagicMock()
    mock.settings = MagicMock()
    mock.settings.tool_budget = 15
    return mock


# ============ Test Fixtures ============


@pytest.fixture
def mock_orchestrator():
    """Fixture providing a mock orchestrator."""
    return create_mock_orchestrator()


@pytest.fixture
def state_modification_workflow():
    """Workflow that modifies state in multiple ways."""

    def append_value(ctx: Dict[str, Any]) -> Dict[str, Any]:
        values = ctx.get("values", [])
        values.append(f"step_{len(values) + 1}")
        ctx["values"] = values
        return ctx

    def increment_counter(ctx: Dict[str, Any]) -> Dict[str, Any]:
        ctx["counter"] = ctx.get("counter", 0) + 1
        return ctx

    def update_metadata(ctx: Dict[str, Any]) -> Dict[str, Any]:
        metadata = ctx.get("metadata", {})
        metadata["last_step"] = ctx.get("values", [])[-1] if ctx.get("values") else None
        metadata["counter_value"] = ctx.get("counter", 0)
        ctx["metadata"] = metadata
        return ctx

    return (
        WorkflowBuilder("state_modification_workflow")
        .add_transform("append1", append_value, next_nodes=["increment1"])
        .add_transform("increment1", increment_counter, next_nodes=["append2"])
        .add_transform("append2", append_value, next_nodes=["increment2"])
        .add_transform("increment2", increment_counter, next_nodes=["update_meta"])
        .add_transform("update_meta", update_metadata)
        .build()
    )


@pytest.fixture
def nested_state_workflow():
    """Workflow that works with deeply nested state."""

    def init_nested(ctx: Dict[str, Any]) -> Dict[str, Any]:
        ctx["nested"] = {"level1": {"level2": {"level3": {"value": "initial", "items": []}}}}
        return ctx

    def modify_nested(ctx: Dict[str, Any]) -> Dict[str, Any]:
        nested = ctx.get("nested", {})
        nested["level1"]["level2"]["level3"]["value"] = "modified"
        nested["level1"]["level2"]["level3"]["items"].append("item1")
        ctx["nested"] = nested
        return ctx

    def add_sibling(ctx: Dict[str, Any]) -> Dict[str, Any]:
        nested = ctx.get("nested", {})
        nested["level1"]["level2"]["level3"]["items"].append("item2")
        nested["level1"]["level2"]["sibling"] = {"added": True}
        ctx["nested"] = nested
        return ctx

    return (
        WorkflowBuilder("nested_state_workflow")
        .add_transform("init", init_nested, next_nodes=["modify"])
        .add_transform("modify", modify_nested, next_nodes=["add_sibling"])
        .add_transform("add_sibling", add_sibling)
        .build()
    )


# ============ Integration Tests ============


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.workflows
class TestStatePersistenceAcrossNodes:
    """Tests for state persistence across workflow nodes."""

    async def test_state_persists_through_linear_workflow(
        self, mock_orchestrator, state_modification_workflow
    ):
        """Test that state modifications persist through linear workflow."""
        executor = WorkflowExecutor(mock_orchestrator)

        result = await executor.execute(state_modification_workflow)

        assert result.success is True

        # Values should accumulate
        values = result.context.data.get("values", [])
        assert len(values) == 2
        assert "step_1" in values
        assert "step_2" in values

        # Counter should increment
        assert result.context.data.get("counter") == 2

        # Metadata should reflect final state
        metadata = result.context.data.get("metadata", {})
        assert metadata.get("last_step") == "step_2"
        assert metadata.get("counter_value") == 2

    async def test_state_from_previous_node_available(self, mock_orchestrator):
        """Test that each node can access state from all previous nodes."""
        state_snapshots = []

        def capture_and_modify_a(ctx: Dict[str, Any]) -> Dict[str, Any]:
            state_snapshots.append(("a", dict(ctx)))
            ctx["from_a"] = True
            ctx["a_value"] = 100
            return ctx

        def capture_and_modify_b(ctx: Dict[str, Any]) -> Dict[str, Any]:
            state_snapshots.append(("b", dict(ctx)))
            ctx["from_b"] = True
            ctx["b_value"] = ctx.get("a_value", 0) * 2
            return ctx

        def capture_and_modify_c(ctx: Dict[str, Any]) -> Dict[str, Any]:
            state_snapshots.append(("c", dict(ctx)))
            ctx["from_c"] = True
            ctx["c_value"] = ctx.get("a_value", 0) + ctx.get("b_value", 0)
            return ctx

        workflow = (
            WorkflowBuilder("capture_workflow")
            .add_transform("a", capture_and_modify_a, next_nodes=["b"])
            .add_transform("b", capture_and_modify_b, next_nodes=["c"])
            .add_transform("c", capture_and_modify_c)
            .build()
        )

        executor = WorkflowExecutor(mock_orchestrator)
        result = await executor.execute(workflow)

        # Node B should see state from A
        b_snapshot = next((s for n, s in state_snapshots if n == "b"), None)
        assert b_snapshot is not None
        assert b_snapshot.get("from_a") is True
        assert b_snapshot.get("a_value") == 100

        # Node C should see state from both A and B
        c_snapshot = next((s for n, s in state_snapshots if n == "c"), None)
        assert c_snapshot is not None
        assert c_snapshot.get("from_a") is True
        assert c_snapshot.get("from_b") is True
        assert c_snapshot.get("a_value") == 100
        assert c_snapshot.get("b_value") == 200

        # Final result should have all values
        assert result.context.data.get("c_value") == 300  # 100 + 200

    async def test_nested_state_persists(self, mock_orchestrator, nested_state_workflow):
        """Test that deeply nested state modifications persist."""
        executor = WorkflowExecutor(mock_orchestrator)

        result = await executor.execute(nested_state_workflow)

        assert result.success is True

        # Check nested modifications
        nested = result.context.data.get("nested", {})
        level3 = nested.get("level1", {}).get("level2", {}).get("level3", {})

        assert level3.get("value") == "modified"
        assert level3.get("items") == ["item1", "item2"]

        # Check sibling addition
        sibling = nested.get("level1", {}).get("level2", {}).get("sibling", {})
        assert sibling.get("added") is True


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.workflows
class TestStateModificationPropagation:
    """Tests for state modification propagation."""

    async def test_list_modifications_propagate(self, mock_orchestrator):
        """Test that list modifications are properly propagated."""

        def add_items(ctx: Dict[str, Any]) -> Dict[str, Any]:
            items = ctx.get("items", [])
            items.extend([{"id": 1}, {"id": 2}])
            ctx["items"] = items
            return ctx

        def modify_items(ctx: Dict[str, Any]) -> Dict[str, Any]:
            items = ctx.get("items", [])
            for item in items:
                item["processed"] = True
            ctx["items"] = items
            return ctx

        def filter_items(ctx: Dict[str, Any]) -> Dict[str, Any]:
            items = ctx.get("items", [])
            ctx["items"] = [i for i in items if i.get("processed")]
            ctx["filtered_count"] = len(ctx["items"])
            return ctx

        workflow = (
            WorkflowBuilder("list_propagation_workflow")
            .add_transform("add", add_items, next_nodes=["modify"])
            .add_transform("modify", modify_items, next_nodes=["filter"])
            .add_transform("filter", filter_items)
            .build()
        )

        executor = WorkflowExecutor(mock_orchestrator)
        result = await executor.execute(workflow)

        assert result.success is True
        assert result.context.data.get("filtered_count") == 2

        items = result.context.data.get("items", [])
        assert all(item.get("processed") for item in items)

    async def test_dict_modifications_propagate(self, mock_orchestrator):
        """Test that dictionary modifications are properly propagated."""

        def set_config(ctx: Dict[str, Any]) -> Dict[str, Any]:
            ctx["config"] = {"version": "1.0", "settings": {"debug": True}}
            return ctx

        def update_config(ctx: Dict[str, Any]) -> Dict[str, Any]:
            config = ctx.get("config", {})
            config["settings"]["logging"] = {"level": "INFO"}
            config["version"] = "1.1"
            ctx["config"] = config
            return ctx

        def read_config(ctx: Dict[str, Any]) -> Dict[str, Any]:
            config = ctx.get("config", {})
            ctx["final_version"] = config.get("version")
            ctx["has_logging"] = "logging" in config.get("settings", {})
            return ctx

        workflow = (
            WorkflowBuilder("dict_propagation_workflow")
            .add_transform("set", set_config, next_nodes=["update"])
            .add_transform("update", update_config, next_nodes=["read"])
            .add_transform("read", read_config)
            .build()
        )

        executor = WorkflowExecutor(mock_orchestrator)
        result = await executor.execute(workflow)

        assert result.success is True
        assert result.context.data.get("final_version") == "1.1"
        assert result.context.data.get("has_logging") is True

    async def test_state_replacement_vs_modification(self, mock_orchestrator):
        """Test that both state replacement and modification work correctly."""

        def replace_state(ctx: Dict[str, Any]) -> Dict[str, Any]:
            # Replace entire value
            ctx["data"] = {"replaced": True}
            return ctx

        def modify_state(ctx: Dict[str, Any]) -> Dict[str, Any]:
            # Modify existing value
            ctx["data"]["modified"] = True
            return ctx

        def add_to_state(ctx: Dict[str, Any]) -> Dict[str, Any]:
            # Add new key
            ctx["new_key"] = "added"
            return ctx

        workflow = (
            WorkflowBuilder("replacement_workflow")
            .add_transform("replace", replace_state, next_nodes=["modify"])
            .add_transform("modify", modify_state, next_nodes=["add"])
            .add_transform("add", add_to_state)
            .build()
        )

        executor = WorkflowExecutor(mock_orchestrator)
        result = await executor.execute(workflow)

        data = result.context.data.get("data", {})
        assert data.get("replaced") is True
        assert data.get("modified") is True
        assert result.context.data.get("new_key") == "added"


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.workflows
class TestLargeStateHandling:
    """Tests for handling large state data."""

    async def test_large_list_state(self, mock_orchestrator):
        """Test handling state with large lists."""

        def create_large_list(ctx: Dict[str, Any]) -> Dict[str, Any]:
            # Create a list with 10,000 items
            ctx["large_list"] = [{"id": i, "data": f"item_{i}"} for i in range(10000)]
            return ctx

        def process_large_list(ctx: Dict[str, Any]) -> Dict[str, Any]:
            large_list = ctx.get("large_list", [])
            ctx["processed_count"] = len(large_list)
            ctx["sum_ids"] = sum(item["id"] for item in large_list)
            return ctx

        workflow = (
            WorkflowBuilder("large_list_workflow")
            .add_transform("create", create_large_list, next_nodes=["process"])
            .add_transform("process", process_large_list)
            .build()
        )

        executor = WorkflowExecutor(mock_orchestrator)
        result = await executor.execute(workflow)

        assert result.success is True
        assert result.context.data.get("processed_count") == 10000
        assert result.context.data.get("sum_ids") == sum(range(10000))

    async def test_large_nested_state(self, mock_orchestrator):
        """Test handling deeply nested state structures."""

        def create_nested_structure(ctx: Dict[str, Any]) -> Dict[str, Any]:
            # Create nested structure with depth
            nested = {}
            current = nested
            for i in range(50):  # 50 levels deep
                current[f"level_{i}"] = {"value": i}
                if i < 49:
                    current[f"level_{i}"]["child"] = {}
                    current = current[f"level_{i}"]["child"]
            ctx["nested"] = nested
            return ctx

        def verify_nested(ctx: Dict[str, Any]) -> Dict[str, Any]:
            # Traverse and verify
            nested = ctx.get("nested", {})
            depth = 0
            current = nested
            while current:
                key = f"level_{depth}"
                if key in current:
                    depth += 1
                    current = current[key].get("child", {})
                else:
                    break
            ctx["verified_depth"] = depth
            return ctx

        workflow = (
            WorkflowBuilder("nested_workflow")
            .add_transform("create", create_nested_structure, next_nodes=["verify"])
            .add_transform("verify", verify_nested)
            .build()
        )

        executor = WorkflowExecutor(mock_orchestrator)
        result = await executor.execute(workflow)

        assert result.success is True
        assert result.context.data.get("verified_depth") == 50

    async def test_large_string_state(self, mock_orchestrator):
        """Test handling large string values in state."""

        def create_large_strings(ctx: Dict[str, Any]) -> Dict[str, Any]:
            # Create large string data
            ctx["large_string"] = "x" * 1_000_000  # 1MB string
            ctx["string_list"] = ["y" * 10000 for _ in range(100)]  # 100 x 10KB strings
            return ctx

        def measure_strings(ctx: Dict[str, Any]) -> Dict[str, Any]:
            ctx["large_string_len"] = len(ctx.get("large_string", ""))
            ctx["total_list_len"] = sum(len(s) for s in ctx.get("string_list", []))
            return ctx

        workflow = (
            WorkflowBuilder("large_string_workflow")
            .add_transform("create", create_large_strings, next_nodes=["measure"])
            .add_transform("measure", measure_strings)
            .build()
        )

        executor = WorkflowExecutor(mock_orchestrator)
        result = await executor.execute(workflow)

        assert result.success is True
        assert result.context.data.get("large_string_len") == 1_000_000
        assert result.context.data.get("total_list_len") == 1_000_000  # 100 * 10000

    async def test_state_with_many_keys(self, mock_orchestrator):
        """Test handling state with many top-level keys."""

        def create_many_keys(ctx: Dict[str, Any]) -> Dict[str, Any]:
            for i in range(1000):
                ctx[f"key_{i}"] = {"index": i, "value": f"value_{i}"}
            return ctx

        def verify_keys(ctx: Dict[str, Any]) -> Dict[str, Any]:
            key_count = sum(1 for k in ctx.keys() if k.startswith("key_"))
            ctx["key_count"] = key_count
            return ctx

        workflow = (
            WorkflowBuilder("many_keys_workflow")
            .add_transform("create", create_many_keys, next_nodes=["verify"])
            .add_transform("verify", verify_keys)
            .build()
        )

        executor = WorkflowExecutor(mock_orchestrator)
        result = await executor.execute(workflow)

        assert result.success is True
        assert result.context.data.get("key_count") == 1000


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.workflows
class TestStateIsolation:
    """Tests for state isolation between workflow instances."""

    async def test_concurrent_workflows_isolated(self, mock_orchestrator):
        """Test that concurrent workflow executions have isolated state."""

        def set_instance_data(ctx: Dict[str, Any]) -> Dict[str, Any]:
            instance_id = ctx.get("instance_id", "unknown")
            ctx["data"] = f"data_from_{instance_id}"
            return ctx

        def verify_instance_data(ctx: Dict[str, Any]) -> Dict[str, Any]:
            instance_id = ctx.get("instance_id", "unknown")
            expected = f"data_from_{instance_id}"
            ctx["verified"] = ctx.get("data") == expected
            return ctx

        workflow = (
            WorkflowBuilder("isolation_workflow")
            .add_transform("set", set_instance_data, next_nodes=["verify"])
            .add_transform("verify", verify_instance_data)
            .build()
        )

        # Run multiple instances concurrently
        executor = WorkflowExecutor(mock_orchestrator)

        results = await asyncio.gather(
            executor.execute(workflow, initial_context={"instance_id": "inst_1"}),
            executor.execute(workflow, initial_context={"instance_id": "inst_2"}),
            executor.execute(workflow, initial_context={"instance_id": "inst_3"}),
        )

        # Each instance should have its own isolated state
        for i, result in enumerate(results):
            assert result.success is True
            assert result.context.data.get("verified") is True
            assert result.context.data.get("instance_id") == f"inst_{i + 1}"
            assert result.context.data.get("data") == f"data_from_inst_{i + 1}"

    async def test_sequential_workflows_isolated(self, mock_orchestrator):
        """Test that sequential workflow executions don't leak state."""

        def append_to_list(ctx: Dict[str, Any]) -> Dict[str, Any]:
            items = ctx.get("items", [])
            items.append(ctx.get("value", "unknown"))
            ctx["items"] = items
            return ctx

        workflow = (
            WorkflowBuilder("sequential_isolation_workflow")
            .add_transform("append", append_to_list)
            .build()
        )

        executor = WorkflowExecutor(mock_orchestrator)

        # Run first workflow
        result1 = await executor.execute(workflow, initial_context={"value": "first", "items": []})

        # Run second workflow
        result2 = await executor.execute(workflow, initial_context={"value": "second", "items": []})

        # Each should only have its own value
        assert result1.context.data.get("items") == ["first"]
        assert result2.context.data.get("items") == ["second"]


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.workflows
class TestWorkflowContextClass:
    """Tests specifically for WorkflowContext functionality."""

    async def test_context_get_set_operations(self, mock_orchestrator):
        """Test WorkflowContext get/set operations during execution."""
        context_operations = []

        def test_context_ops(ctx: Dict[str, Any]) -> Dict[str, Any]:
            # These operations simulate what happens inside WorkflowContext
            ctx["test_key"] = "test_value"
            context_operations.append(("set", "test_key", "test_value"))

            retrieved = ctx.get("test_key", "default")
            context_operations.append(("get", "test_key", retrieved))

            missing = ctx.get("missing_key", "default")
            context_operations.append(("get_default", "missing_key", missing))

            return ctx

        workflow = (
            WorkflowBuilder("context_ops_workflow").add_transform("test", test_context_ops).build()
        )

        executor = WorkflowExecutor(mock_orchestrator)
        result = await executor.execute(workflow)

        assert result.success is True
        assert result.context.data.get("test_key") == "test_value"

    async def test_context_update_multiple_values(self, mock_orchestrator):
        """Test updating multiple context values at once."""

        def batch_update(ctx: Dict[str, Any]) -> Dict[str, Any]:
            updates = {
                "key1": "value1",
                "key2": "value2",
                "key3": "value3",
            }
            ctx.update(updates)
            return ctx

        def verify_update(ctx: Dict[str, Any]) -> Dict[str, Any]:
            ctx["all_present"] = all(ctx.get(f"key{i}") == f"value{i}" for i in range(1, 4))
            return ctx

        workflow = (
            WorkflowBuilder("batch_update_workflow")
            .add_transform("update", batch_update, next_nodes=["verify"])
            .add_transform("verify", verify_update)
            .build()
        )

        executor = WorkflowExecutor(mock_orchestrator)
        result = await executor.execute(workflow)

        assert result.success is True
        assert result.context.data.get("all_present") is True

    async def test_context_node_result_tracking(self, mock_orchestrator):
        """Test that context properly tracks node results."""

        def step(ctx: Dict[str, Any]) -> Dict[str, Any]:
            ctx["processed"] = True
            return ctx

        workflow = (
            WorkflowBuilder("tracking_workflow")
            .add_transform("step1", step, next_nodes=["step2"])
            .add_transform("step2", step)
            .build()
        )

        executor = WorkflowExecutor(mock_orchestrator)
        result = await executor.execute(workflow)

        # Node results should be tracked
        assert "step1" in result.context.node_results or "step2" in result.context.node_results

    async def test_context_has_failures_detection(self, mock_orchestrator):
        """Test WorkflowContext.has_failures() detection."""
        context = WorkflowContext()

        # Initially no failures
        assert context.has_failures() is False

        # Add a successful result
        context.add_result(
            NodeResult(
                node_id="success_node",
                status=ExecutorNodeStatus.COMPLETED,
            )
        )
        assert context.has_failures() is False

        # Add a failed result
        context.add_result(
            NodeResult(
                node_id="failed_node",
                status=ExecutorNodeStatus.FAILED,
                error="Test error",
            )
        )
        assert context.has_failures() is True

    async def test_context_get_outputs(self, mock_orchestrator):
        """Test WorkflowContext.get_outputs() functionality."""
        context = WorkflowContext()

        # Add results with outputs
        context.add_result(
            NodeResult(
                node_id="node1",
                status=ExecutorNodeStatus.COMPLETED,
                output={"data": "output1"},
            )
        )
        context.add_result(
            NodeResult(
                node_id="node2",
                status=ExecutorNodeStatus.COMPLETED,
                output={"data": "output2"},
            )
        )
        context.add_result(
            NodeResult(
                node_id="failed_node",
                status=ExecutorNodeStatus.FAILED,
                output=None,
            )
        )

        outputs = context.get_outputs()

        # Should only include successful nodes with outputs
        assert "node1" in outputs
        assert "node2" in outputs
        assert "failed_node" not in outputs


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.workflows
class TestStateSerializationEdgeCases:
    """Tests for state serialization edge cases."""

    async def test_state_with_none_values(self, mock_orchestrator):
        """Test handling state with None values."""

        def set_nones(ctx: Dict[str, Any]) -> Dict[str, Any]:
            ctx["none_value"] = None
            ctx["list_with_nones"] = [1, None, 3, None]
            ctx["dict_with_nones"] = {"key1": None, "key2": "value"}
            return ctx

        def verify_nones(ctx: Dict[str, Any]) -> Dict[str, Any]:
            ctx["none_preserved"] = ctx.get("none_value") is None
            ctx["list_nones_preserved"] = ctx.get("list_with_nones") == [
                1,
                None,
                3,
                None,
            ]
            return ctx

        workflow = (
            WorkflowBuilder("nones_workflow")
            .add_transform("set", set_nones, next_nodes=["verify"])
            .add_transform("verify", verify_nones)
            .build()
        )

        executor = WorkflowExecutor(mock_orchestrator)
        result = await executor.execute(workflow)

        assert result.success is True
        assert result.context.data.get("none_preserved") is True
        assert result.context.data.get("list_nones_preserved") is True

    async def test_state_with_boolean_values(self, mock_orchestrator):
        """Test handling state with boolean values."""

        def set_booleans(ctx: Dict[str, Any]) -> Dict[str, Any]:
            ctx["true_val"] = True
            ctx["false_val"] = False
            ctx["bool_list"] = [True, False, True]
            return ctx

        def verify_booleans(ctx: Dict[str, Any]) -> Dict[str, Any]:
            ctx["true_is_true"] = ctx.get("true_val") is True
            ctx["false_is_false"] = ctx.get("false_val") is False
            ctx["list_correct"] = ctx.get("bool_list") == [True, False, True]
            return ctx

        workflow = (
            WorkflowBuilder("booleans_workflow")
            .add_transform("set", set_booleans, next_nodes=["verify"])
            .add_transform("verify", verify_booleans)
            .build()
        )

        executor = WorkflowExecutor(mock_orchestrator)
        result = await executor.execute(workflow)

        assert result.success is True
        assert result.context.data.get("true_is_true") is True
        assert result.context.data.get("false_is_false") is True
        assert result.context.data.get("list_correct") is True

    async def test_state_with_numeric_types(self, mock_orchestrator):
        """Test handling state with various numeric types."""

        def set_numbers(ctx: Dict[str, Any]) -> Dict[str, Any]:
            ctx["integer"] = 42
            ctx["float_val"] = 3.14159
            ctx["negative"] = -100
            ctx["large_int"] = 10**20
            ctx["zero"] = 0
            return ctx

        def verify_numbers(ctx: Dict[str, Any]) -> Dict[str, Any]:
            ctx["integer_correct"] = ctx.get("integer") == 42
            ctx["float_approx"] = abs(ctx.get("float_val", 0) - 3.14159) < 0.0001
            ctx["negative_correct"] = ctx.get("negative") == -100
            ctx["large_correct"] = ctx.get("large_int") == 10**20
            ctx["zero_correct"] = ctx.get("zero") == 0
            return ctx

        workflow = (
            WorkflowBuilder("numbers_workflow")
            .add_transform("set", set_numbers, next_nodes=["verify"])
            .add_transform("verify", verify_numbers)
            .build()
        )

        executor = WorkflowExecutor(mock_orchestrator)
        result = await executor.execute(workflow)

        assert result.success is True
        assert result.context.data.get("integer_correct") is True
        assert result.context.data.get("float_approx") is True
        assert result.context.data.get("negative_correct") is True
        assert result.context.data.get("large_correct") is True
        assert result.context.data.get("zero_correct") is True

    async def test_state_with_empty_collections(self, mock_orchestrator):
        """Test handling state with empty collections."""

        def set_empties(ctx: Dict[str, Any]) -> Dict[str, Any]:
            ctx["empty_list"] = []
            ctx["empty_dict"] = {}
            ctx["empty_string"] = ""
            return ctx

        def verify_empties(ctx: Dict[str, Any]) -> Dict[str, Any]:
            ctx["list_empty"] = ctx.get("empty_list") == []
            ctx["dict_empty"] = ctx.get("empty_dict") == {}
            ctx["string_empty"] = ctx.get("empty_string") == ""
            return ctx

        workflow = (
            WorkflowBuilder("empties_workflow")
            .add_transform("set", set_empties, next_nodes=["verify"])
            .add_transform("verify", verify_empties)
            .build()
        )

        executor = WorkflowExecutor(mock_orchestrator)
        result = await executor.execute(workflow)

        assert result.success is True
        assert result.context.data.get("list_empty") is True
        assert result.context.data.get("dict_empty") is True
        assert result.context.data.get("string_empty") is True
