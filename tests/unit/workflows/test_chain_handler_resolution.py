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

"""Unit tests for chain handler resolution in compute nodes.

TDD approach: Test each method and edge case in isolation.
"""

from __future__ import annotations

import asyncio
import pytest
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

from victor.workflows.executors.compute import ComputeNodeExecutor
from victor.workflows.definition import ComputeNode, TaskConstraints
from victor.workflows.context import WorkflowContext

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def reset_chain_registry():
    """Reset chain registry before each test."""
    from victor.framework.chain_registry import reset_chain_registry

    reset_chain_registry()
    yield
    reset_chain_registry()


@pytest.fixture
def executor():
    """Create a ComputeNodeExecutor for testing."""
    return ComputeNodeExecutor()


@pytest.fixture
def mock_context():
    """Create a mock WorkflowContext."""
    return WorkflowContext({"test_key": "test_value", "number": 42})


# =============================================================================
# Unit Tests: _get_compute_handler
# =============================================================================


class TestGetComputeHandler:
    """Unit tests for _get_compute_handler method."""

    def test_chain_prefix_routing(self, executor):
        """Chain: prefix routes to _resolve_chain_handler."""
        with patch.object(executor, "_resolve_chain_handler") as mock_resolve:
            mock_resolve.return_value = "mocked_handler"
            result = executor._get_compute_handler("chain:vertical:name")
            assert result == "mocked_handler"
            mock_resolve.assert_called_once_with("chain:vertical:name")

    def test_regular_handler_uses_registry(self, executor):
        """Non-chain: prefix uses get_compute_handler."""
        # Note: get_compute_handler is imported inside _get_compute_handler,
        # so we patch it at its source location
        with patch("victor.workflows.compute_registry.get_compute_handler") as mock_get:
            mock_get.return_value = "regular_handler"
            result = executor._get_compute_handler("regular_handler")
            assert result == "regular_handler"
            mock_get.assert_called_once_with("regular_handler")

    def test_empty_string_handler(self, executor):
        """Empty string handler returns None."""
        result = executor._get_compute_handler("")
        assert result is None

    def test_none_handler(self, executor):
        """None handler raises appropriate error."""
        with pytest.raises(AttributeError):
            executor._get_compute_handler(None)


# =============================================================================
# Unit Tests: _resolve_chain_handler
# =============================================================================


class TestResolveChainHandler:
    """Unit tests for _resolve_chain_handler method."""

    def test_parse_vertical_name_format(self, executor, reset_chain_registry):
        """Parse chain:vertical:name format correctly."""
        from victor.framework.chain_registry import chain
        from victor.tools.composition import RunnableLambda

        @chain("test:my_chain")
        def my_chain():
            return RunnableLambda(lambda x: x)

        result = executor._resolve_chain_handler("chain:test:my_chain")
        assert result is not None
        assert callable(result)

    def test_parse_name_only_format(self, executor, reset_chain_registry):
        """Parse chain:name format (no vertical) correctly."""
        from victor.framework.chain_registry import chain
        from victor.tools.composition import RunnableLambda

        @chain("standalone")
        def standalone_chain():
            return RunnableLambda(lambda x: x)

        result = executor._resolve_chain_handler("chain:standalone")
        assert result is not None
        assert callable(result)

    def test_chain_not_found_returns_none(self, executor, reset_chain_registry):
        """Non-existent chain returns None gracefully."""
        result = executor._resolve_chain_handler("chain:nonexistent:missing")
        assert result is None

    def test_empty_chain_reference(self, executor, reset_chain_registry):
        """Empty chain reference (chain:) returns None."""
        result = executor._resolve_chain_handler("chain:")
        assert result is None

    def test_multiple_colons_in_reference(self, executor, reset_chain_registry):
        """Handle multiple colons (chain:a:b:c) - treats first as vertical."""
        from victor.framework.chain_registry import chain
        from victor.tools.composition import RunnableLambda

        # Register with vertical "a" and name "b:c"
        @chain("a:b:c")
        def complex_chain():
            return RunnableLambda(lambda x: x)

        result = executor._resolve_chain_handler("chain:a:b:c")
        assert result is not None


# =============================================================================
# Unit Tests: _create_chain_wrapper
# =============================================================================


class TestCreateChainWrapper:
    """Unit tests for _create_chain_wrapper method."""

    @pytest.mark.asyncio
    async def test_wrapper_returns_async_function(self, executor):
        """Wrapper returns an async callable."""
        from victor.tools.composition import RunnableLambda

        runnable = RunnableLambda(lambda x: {"result": x["input"] * 2})
        wrapper = executor._create_chain_wrapper(runnable, "test_chain")

        assert asyncio.iscoroutinefunction(wrapper)

    @pytest.mark.asyncio
    async def test_wrapper_executes_runnable(self, executor, mock_context):
        """Wrapper executes the runnable and returns NodeResult."""
        from victor.tools.composition import RunnableLambda
        from victor.workflows.definition import ComputeNode

        runnable = RunnableLambda(lambda x: {"result": x.get("value", 0) * 2})
        wrapper = executor._create_chain_wrapper(runnable, "test_chain")

        node = ComputeNode(
            id="test_node",
            name="Test Node",
            handler="chain:test_chain",
            input_mapping={"value": "$ctx.number"},
        )

        result = await wrapper(node, mock_context, None)

        assert result.status.value == "completed"
        assert result.output["result"] == 84  # 42 * 2

    @pytest.mark.asyncio
    async def test_wrapper_handles_timeout(self, executor, mock_context):
        """Wrapper respects timeout from node constraints."""
        from victor.tools.composition import RunnableLambda
        from victor.workflows.definition import ComputeNode

        async def slow_fn(x):
            await asyncio.sleep(5)
            return {"done": True}

        runnable = RunnableLambda(slow_fn)
        wrapper = executor._create_chain_wrapper(runnable, "slow_chain")

        node = ComputeNode(
            id="timeout_node",
            name="Timeout Node",
            handler="chain:slow_chain",
            constraints=TaskConstraints(_timeout=0.5),
        )

        result = await wrapper(node, mock_context, None)

        assert result.status.value == "failed"
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_wrapper_propagates_exceptions(self, executor, mock_context):
        """Wrapper converts exceptions to failed NodeResult."""
        from victor.tools.composition import RunnableLambda
        from victor.workflows.definition import ComputeNode

        def failing_fn(x):
            raise ValueError("Simulated failure")

        runnable = RunnableLambda(failing_fn)
        wrapper = executor._create_chain_wrapper(runnable, "failing_chain")

        node = ComputeNode(
            id="error_node",
            name="Error Node",
            handler="chain:failing_chain",
        )

        result = await wrapper(node, mock_context, None)

        assert result.status.value == "failed"
        assert "execution failed" in result.error.lower()
        assert "Simulated failure" in result.error


# =============================================================================
# Unit Tests: _prepare_chain_input
# =============================================================================


class TestPrepareChainInput:
    """Unit tests for _prepare_chain_input method."""

    def test_input_mapping_with_ctx_prefix(self, executor, mock_context):
        """Map $ctx. prefix to context values."""
        from victor.workflows.definition import ComputeNode

        node = ComputeNode(
            id="test_node",
            name="Test Node",
            input_mapping={"param1": "$ctx.test_key", "param2": "$ctx.number"},
        )

        input_data = executor._prepare_chain_input(node, mock_context)

        assert input_data["param1"] == "test_value"
        assert input_data["param2"] == 42

    def test_input_mapping_with_state_prefix(self, executor, mock_context):
        """Map $state. prefix to context values (same as $ctx.)."""
        from victor.workflows.definition import ComputeNode

        node = ComputeNode(
            id="test_node",
            name="Test Node",
            input_mapping={"param1": "$state.test_key"},
        )

        input_data = executor._prepare_chain_input(node, mock_context)

        assert input_data["param1"] == "test_value"

    def test_input_mapping_missing_key_returns_key_name(self, executor, mock_context):
        """Missing context keys return the key name as fallback."""
        from victor.workflows.definition import ComputeNode

        node = ComputeNode(
            id="test_node",
            name="Test Node",
            input_mapping={"param1": "$ctx.nonexistent"},
        )

        input_data = executor._prepare_chain_input(node, mock_context)

        assert input_data["param1"] == "nonexistent"

    def test_input_mapping_with_direct_values(self, executor, mock_context):
        """Direct values (non-$ctx. strings) are used as-is."""
        from victor.workflows.definition import ComputeNode

        node = ComputeNode(
            id="test_node",
            name="Test Node",
            input_mapping={"param1": "literal_value", "param2": 123},
        )

        input_data = executor._prepare_chain_input(node, mock_context)

        assert input_data["param1"] == "literal_value"
        assert input_data["param2"] == 123

    def test_no_input_mapping_uses_all_context(self, executor, mock_context):
        """Without input_mapping, all context data is passed."""
        from victor.workflows.definition import ComputeNode

        node = ComputeNode(
            id="test_node",
            name="Test Node",
        )

        input_data = executor._prepare_chain_input(node, mock_context)

        assert input_data == {"test_key": "test_value", "number": 42}

    def test_empty_context_no_mapping(self, executor):
        """Empty context with no mapping returns empty dict."""
        from victor.workflows.definition import ComputeNode

        node = ComputeNode(
            id="test_node",
            name="Test Node",
        )
        context = WorkflowContext({})

        input_data = executor._prepare_chain_input(node, context)

        assert input_data == {}


# =============================================================================
# Unit Tests: _update_context
# =============================================================================


class TestUpdateContext:
    """Unit tests for _update_context method."""

    def test_update_with_output_key(self, executor, mock_context):
        """Update context with output_key."""
        from victor.workflows.definition import ComputeNode

        node = ComputeNode(
            id="test_node",
            name="Test Node",
            output_key="result",
        )

        output = {"score": 95, "issues": []}
        executor._update_context(mock_context, node, output)

        assert mock_context.get("result") == output

    def test_update_without_output_key_merges_dict(self, executor, mock_context):
        """Without output_key, merge dict output to context."""
        from victor.workflows.definition import ComputeNode

        node = ComputeNode(
            id="test_node",
            name="Test Node",
        )

        output = {"score": 95, "issues": [], "_internal": "hidden"}
        executor._update_context(mock_context, node, output)

        assert mock_context.get("score") == 95
        assert mock_context.get("issues") == []
        # Private keys starting with _ should not be set
        assert mock_context.get("_internal") is None

    def test_update_with_non_dict_output_ignored(self, executor, mock_context):
        """Non-dict output is ignored without output_key."""
        from victor.workflows.definition import ComputeNode

        node = ComputeNode(
            id="test_node",
            name="Test Node",
        )

        executor._update_context(mock_context, node, "string_output")

        # Context should remain unchanged
        assert mock_context.data == {"test_key": "test_value", "number": 42}


# =============================================================================
# Unit Tests: State Isolation
# =============================================================================


class TestStateIsolation:
    """Unit tests for state isolation between chain executions."""

    @pytest.mark.asyncio
    async def test_context_changes_isolated_between_executions(
        self, executor, reset_chain_registry
    ):
        """Changes during chain execution don't affect original context until merged."""
        from victor.framework.chain_registry import chain
        from victor.tools.composition import RunnableLambda
        from victor.workflows.definition import ComputeNode

        @chain("isolation_test:add")
        def add_chain():
            return RunnableLambda(lambda x: {"result": x["value"] + 10})

        @chain("isolation_test:multiply")
        def multiply_chain():
            return RunnableLambda(lambda x: {"result": x["value"] * 2})

        context = WorkflowContext({"value": 5})

        # Execute first chain
        handler1 = executor._resolve_chain_handler("chain:isolation_test:add")
        node1 = ComputeNode(
            id="add_node",
            name="Add Node",
            handler="chain:isolation_test:add",
            output_key="add_result",
        )
        result1 = await handler1(node1, context, None)

        assert result1.output["result"] == 15
        assert context.get("add_result") == {"result": 15}
        # Original value should still be intact
        assert context.get("value") == 5

        # Execute second chain with same original value
        handler2 = executor._resolve_chain_handler("chain:isolation_test:multiply")
        node2 = ComputeNode(
            id="multiply_node",
            name="Multiply Node",
            handler="chain:isolation_test:multiply",
            input_mapping={"value": "$ctx.value"},
            output_key="multiply_result",
        )
        result2 = await handler2(node2, context, None)

        assert result2.output["result"] == 10  # 5 * 2, not 15 * 2


# =============================================================================
# Unit Tests: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Unit tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_chain_returns_none(self, executor, reset_chain_registry, mock_context):
        """Handle chains that return None."""
        from victor.framework.chain_registry import chain
        from victor.tools.composition import RunnableLambda
        from victor.workflows.definition import ComputeNode

        @chain("none_test:null")
        def null_chain():
            return RunnableLambda(lambda x: None)

        handler = executor._resolve_chain_handler("chain:none_test:null")
        node = ComputeNode(
            id="null_node",
            name="Null Node",
            handler="chain:none_test:null",
        )

        result = await handler(node, mock_context, None)

        assert result.status.value == "completed"
        assert result.output is None

    @pytest.mark.asyncio
    async def test_chain_returns_non_dict(self, executor, reset_chain_registry, mock_context):
        """Handle chains that return non-dict values."""
        from victor.framework.chain_registry import chain
        from victor.tools.composition import RunnableLambda
        from victor.workflows.definition import ComputeNode

        @chain("type_test:string")
        def string_chain():
            return RunnableLambda(lambda x: "just_a_string")

        handler = executor._resolve_chain_handler("chain:type_test:string")
        node = ComputeNode(
            id="string_node",
            name="String Node",
            handler="chain:type_test:string",
            output_key="string_result",
        )

        result = await handler(node, mock_context, None)

        assert result.status.value == "completed"
        assert result.output == "just_a_string"
        assert mock_context.get("string_result") == "just_a_string"

    @pytest.mark.asyncio
    async def test_chain_returns_complex_nested_structure(
        self, executor, reset_chain_registry, mock_context
    ):
        """Handle chains that return complex nested data structures."""
        from victor.framework.chain_registry import chain
        from victor.tools.composition import RunnableLambda
        from victor.workflows.definition import ComputeNode

        @chain("complex_test:nested")
        def nested_chain():
            return RunnableLambda(
                lambda x: {
                    "level1": {
                        "level2": {
                            "level3": ["a", "b", "c"],
                            "count": 3,
                        },
                        "metadata": {"version": 1.0},
                    },
                    "top_level": "value",
                }
            )

        handler = executor._resolve_chain_handler("chain:complex_test:nested")
        node = ComputeNode(
            id="nested_node",
            name="Nested Node",
            handler="chain:complex_test:nested",
        )

        result = await handler(node, mock_context, None)

        assert result.status.value == "completed"
        assert result.output["level1"]["level2"]["level3"] == ["a", "b", "c"]
        assert result.output["top_level"] == "value"

    @pytest.mark.asyncio
    async def test_chain_with_empty_input(self, executor, reset_chain_registry):
        """Handle chains called with empty input."""
        from victor.framework.chain_registry import chain
        from victor.tools.composition import RunnableLambda
        from victor.workflows.definition import ComputeNode

        @chain("empty_test:process")
        def process_chain():
            return RunnableLambda(lambda x: {"processed": True, "input_keys": list(x.keys())})

        handler = executor._resolve_chain_handler("chain:empty_test:process")
        node = ComputeNode(
            id="empty_node",
            name="Empty Node",
            handler="chain:empty_test:process",
        )
        context = WorkflowContext({})

        result = await handler(node, context, None)

        assert result.status.value == "completed"
        assert result.output["processed"] is True
        assert result.output["input_keys"] == []


# =============================================================================
# Integration Tests: Full Workflow
# =============================================================================


class TestFullWorkflowIntegration:
    """Integration tests for chain handlers in complete workflows."""

    @pytest.mark.asyncio
    async def test_workflow_with_chain_handler(self, reset_chain_registry):
        """Test complete workflow with chain handler."""
        from victor.framework.chain_registry import chain
        from victor.tools.composition import RunnableLambda
        from victor.workflows.definition import ComputeNode
        from victor.workflows.context import WorkflowContext

        @chain("workflow_test:analyze")
        def analyze_chain():
            return RunnableLambda(
                lambda x: {
                    "score": x.get("complexity", 1) * 10,
                    "issues": ["minor"] if x.get("complexity", 1) < 5 else ["major"],
                }
            )

        executor = ComputeNodeExecutor()
        handler = executor._resolve_chain_handler("chain:workflow_test:analyze")

        node = ComputeNode(
            id="analyze",
            name="Analyze",
            handler="chain:workflow_test:analyze",
            input_mapping={"complexity": "$ctx.code_complexity"},
            output_key="analysis",
        )

        context = WorkflowContext({"code_complexity": 3})
        result = await handler(node, context, None)

        assert result.status.value == "completed"
        assert result.output["score"] == 30
        assert result.output["issues"] == ["minor"]
        assert context.get("analysis") == result.output


# =============================================================================
# Tests Summary
# =============================================================================


__all__ = [
    "TestGetComputeHandler",
    "TestResolveChainHandler",
    "TestCreateChainWrapper",
    "TestPrepareChainInput",
    "TestUpdateContext",
    "TestStateIsolation",
    "TestEdgeCases",
    "TestFullWorkflowIntegration",
]
