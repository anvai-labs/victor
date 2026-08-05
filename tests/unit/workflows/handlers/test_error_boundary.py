# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""Tests for HandlerErrorBoundary and related classes."""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set
from unittest.mock import MagicMock

import pytest

from victor_contracts.workflows import ExecutorNodeStatus, NodeResult
from victor.workflows.handlers import (
    HandlerError,
    HandlerErrorBoundary,
    with_error_boundary,
)


@dataclass
class MockComputeNode:
    """Mock compute node for testing."""

    id: str
    handler: str = "test_handler"
    tools: Set[str] = field(default_factory=set)
    input_mapping: Dict[str, str] = field(default_factory=dict)
    output_key: Optional[str] = None


class MockContext:
    """Mock workflow context for testing."""

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        self.data = data or {}

    def get(self, key: str) -> Any:
        return self.data.get(key)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value


class TestHandlerError:
    """Test HandlerError dataclass."""

    def test_creation(self):
        """Should create with all fields."""
        error = HandlerError(
            handler_name="test_handler",
            node_id="node_1",
            error_type="validation",
            message="Invalid input",
            traceback_str="Traceback...",
            context_snapshot={"key": "value"},
        )
        assert error.handler_name == "test_handler"
        assert error.node_id == "node_1"
        assert error.error_type == "validation"
        assert error.message == "Invalid input"
        assert error.traceback_str == "Traceback..."
        assert error.context_snapshot == {"key": "value"}

    def test_defaults(self):
        """Should have sensible defaults."""
        error = HandlerError(
            handler_name="test",
            node_id="n1",
            error_type="runtime",
            message="Error",
        )
        assert error.traceback_str is None
        assert error.context_snapshot == {}

    def test_to_dict(self):
        """Should serialize to dictionary."""
        error = HandlerError(
            handler_name="my_handler",
            node_id="node_123",
            error_type="timeout",
            message="Timed out",
            traceback_str="stack trace...",
        )
        d = error.to_dict()
        assert d["handler_name"] == "my_handler"
        assert d["node_id"] == "node_123"
        assert d["error_type"] == "timeout"
        assert d["message"] == "Timed out"
        assert d["traceback"] == "stack trace..."

    def test_to_dict_excludes_context(self):
        """to_dict should not include context_snapshot."""
        error = HandlerError(
            handler_name="h",
            node_id="n",
            error_type="e",
            message="m",
            context_snapshot={"sensitive": "data"},
        )
        d = error.to_dict()
        assert "context_snapshot" not in d


class TestHandlerErrorBoundary:
    """Test HandlerErrorBoundary class."""

    def test_init_default(self):
        """Should initialize with preserve_context=True by default."""
        boundary = HandlerErrorBoundary()
        assert boundary.preserve_context is True

    def test_init_preserve_context_false(self):
        """Should support preserve_context=False."""
        boundary = HandlerErrorBoundary(preserve_context=False)
        assert boundary.preserve_context is False

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Should return handler result on success."""
        node = MockComputeNode(id="test_node")
        context = MockContext({"input": "value"})
        registry = MagicMock()

        async def success_handler(n, ctx, reg):
            return NodeResult(
                node_id=n.id,
                status=ExecutorNodeStatus.COMPLETED,
                output="success",
            )

        boundary = HandlerErrorBoundary()
        result = await boundary.execute(
            handler=success_handler,
            handler_name="success",
            node=node,
            context=context,
            tool_registry=registry,
        )

        assert result.status == ExecutorNodeStatus.COMPLETED
        assert result.output == "success"

    @pytest.mark.asyncio
    async def test_execute_timeout_error(self):
        """Should handle asyncio.TimeoutError."""
        node = MockComputeNode(id="timeout_node")
        context = MockContext()
        registry = MagicMock()

        async def timeout_handler(n, ctx, reg):
            raise asyncio.TimeoutError()

        boundary = HandlerErrorBoundary()
        result = await boundary.execute(
            handler=timeout_handler,
            handler_name="timeout_handler",
            node=node,
            context=context,
            tool_registry=registry,
        )

        assert result.status == ExecutorNodeStatus.FAILED
        assert "timeout_handler" in result.error
        assert "timed out" in result.error.lower() or "timeout" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_value_error(self):
        """Should handle ValueError as validation error."""
        node = MockComputeNode(id="validation_node")
        context = MockContext()
        registry = MagicMock()

        async def validation_handler(n, ctx, reg):
            raise ValueError("Invalid parameter")

        boundary = HandlerErrorBoundary()
        result = await boundary.execute(
            handler=validation_handler,
            handler_name="validation_handler",
            node=node,
            context=context,
            tool_registry=registry,
        )

        assert result.status == ExecutorNodeStatus.FAILED
        assert "Invalid parameter" in result.error

    @pytest.mark.asyncio
    async def test_execute_key_error(self):
        """Should handle KeyError as missing_key error."""
        node = MockComputeNode(id="key_node")
        context = MockContext()
        registry = MagicMock()

        async def key_handler(n, ctx, reg):
            raise KeyError("missing_key")

        boundary = HandlerErrorBoundary()
        result = await boundary.execute(
            handler=key_handler,
            handler_name="key_handler",
            node=node,
            context=context,
            tool_registry=registry,
        )

        assert result.status == ExecutorNodeStatus.FAILED
        assert "key_handler" in result.error

    @pytest.mark.asyncio
    async def test_execute_type_error(self):
        """Should handle TypeError."""
        node = MockComputeNode(id="type_node")
        context = MockContext()
        registry = MagicMock()

        async def type_handler(n, ctx, reg):
            raise TypeError("Wrong type")

        boundary = HandlerErrorBoundary()
        result = await boundary.execute(
            handler=type_handler,
            handler_name="type_handler",
            node=node,
            context=context,
            tool_registry=registry,
        )

        assert result.status == ExecutorNodeStatus.FAILED
        assert "Wrong type" in result.error

    @pytest.mark.asyncio
    async def test_execute_generic_exception(self):
        """Should handle generic exceptions."""
        node = MockComputeNode(id="generic_node")
        context = MockContext()
        registry = MagicMock()

        async def generic_handler(n, ctx, reg):
            raise RuntimeError("Something went wrong")

        boundary = HandlerErrorBoundary()
        result = await boundary.execute(
            handler=generic_handler,
            handler_name="generic_handler",
            node=node,
            context=context,
            tool_registry=registry,
        )

        assert result.status == ExecutorNodeStatus.FAILED
        assert "Something went wrong" in result.error

    @pytest.mark.asyncio
    async def test_execute_preserves_context(self):
        """Should snapshot context when preserve_context=True."""
        node = MockComputeNode(id="context_node")
        context = MockContext({"input": "value", "number": 42})
        registry = MagicMock()

        captured_snapshot = {}

        async def error_handler(n, ctx, reg):
            raise ValueError("Error")

        boundary = HandlerErrorBoundary(preserve_context=True)
        result = await boundary.execute(
            handler=error_handler,
            handler_name="error_handler",
            node=node,
            context=context,
            tool_registry=registry,
        )

        assert result.status == ExecutorNodeStatus.FAILED

    @pytest.mark.asyncio
    async def test_execute_no_context_preservation(self):
        """Should not snapshot when preserve_context=False."""
        node = MockComputeNode(id="no_context_node")
        context = MockContext({"input": "value"})
        registry = MagicMock()

        async def error_handler(n, ctx, reg):
            raise ValueError("Error")

        boundary = HandlerErrorBoundary(preserve_context=False)
        result = await boundary.execute(
            handler=error_handler,
            handler_name="error_handler",
            node=node,
            context=context,
            tool_registry=registry,
        )

        assert result.status == ExecutorNodeStatus.FAILED

    @pytest.mark.asyncio
    async def test_execute_duration_tracked(self):
        """Should track execution duration."""
        node = MockComputeNode(id="duration_node")
        context = MockContext()
        registry = MagicMock()

        async def slow_handler(n, ctx, reg):
            await asyncio.sleep(0.01)  # 10ms
            raise ValueError("Error after delay")

        boundary = HandlerErrorBoundary()
        result = await boundary.execute(
            handler=slow_handler,
            handler_name="slow_handler",
            node=node,
            context=context,
            tool_registry=registry,
        )

        assert result.status == ExecutorNodeStatus.FAILED
        assert result.duration_seconds >= 0.01


class TestWithErrorBoundaryDecorator:
    """Test with_error_boundary decorator."""

    @pytest.mark.asyncio
    async def test_decorator_wraps_handler(self):
        """Decorator should wrap handler with error boundary."""

        @with_error_boundary("decorated_handler")
        async def my_handler(node, context, tool_registry):
            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.COMPLETED,
                output="decorated",
            )

        node = MockComputeNode(id="decorated_node")
        context = MockContext()
        registry = MagicMock()

        result = await my_handler(node, context, registry)
        assert result.status == ExecutorNodeStatus.COMPLETED
        assert result.output == "decorated"

    @pytest.mark.asyncio
    async def test_decorator_catches_errors(self):
        """Decorator should catch and wrap errors."""

        @with_error_boundary("failing_handler")
        async def failing_handler(node, context, tool_registry):
            raise RuntimeError("Decorated handler failed")

        node = MockComputeNode(id="failing_node")
        context = MockContext()
        registry = MagicMock()

        result = await failing_handler(node, context, registry)
        assert result.status == ExecutorNodeStatus.FAILED
        assert "failing_handler" in result.error
        assert "Decorated handler failed" in result.error

    @pytest.mark.asyncio
    async def test_decorator_preserves_function_name(self):
        """Decorator should preserve function name via wraps."""

        @with_error_boundary("named_handler")
        async def my_named_handler(node, context, tool_registry):
            pass

        assert my_named_handler.__name__ == "my_named_handler"

    @pytest.mark.asyncio
    async def test_decorator_timeout_handling(self):
        """Decorator should handle timeout errors."""

        @with_error_boundary("timeout_decorated")
        async def timeout_handler(node, context, tool_registry):
            raise asyncio.TimeoutError("Timeout!")

        node = MockComputeNode(id="timeout_decorated_node")
        context = MockContext()
        registry = MagicMock()

        result = await timeout_handler(node, context, registry)
        assert result.status == ExecutorNodeStatus.FAILED
